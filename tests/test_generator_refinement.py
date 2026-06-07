import unittest  # Python's built-in testing framework — provides the TestCase base class and all assertion helpers.
from unittest.mock import patch  # Allows replacing real dependencies (like LLM calls and key checks) with controlled fakes during tests.

# Import the functions and constants from the test_generator agent that we are testing.
from agents.test_generator import (
    INITIAL_GENERATION_CATEGORIES,       # The list of test categories sent to the LLM in Pass 1.
    OUTPUT_TEST_CATEGORIES,              # The authoritative taxonomy of test output categories.
    _build_heuristic_coverage_report,    # Builds a rule-based coverage report without calling the LLM.
    _decorate_test_cases,                # Assigns canonical output categories and enriches test cases post-generation.
    generate_test_cases,                 # The main entry-point function: runs the full generation + refinement loop.
)
# Import supporting data structures from test_suite.py used to set up test scenarios.
from utils.test_suite import (
    CoverageCategoryReport,  # Represents the coverage status for a single category (used to build mock reports).
    CoverageReport,          # The full coverage report object used by the refinement loop.
    TestCase,                # The core data model for a single test case.
    parse_test_cases,        # Parses the final markdown string back into TestCase objects for assertions.
)


# ─────────────────────────────────────────────────────────────────────
# SHARED FIXTURE — a minimal requirement analysis text used as the
# 'context' input for tests that run the full generate_test_cases() loop.
# It simulates what the AI Analyzer would produce for a simple
# user registration and login feature.
# ─────────────────────────────────────────────────────────────────────
SAMPLE_ANALYSIS = """
Features:
- Users can register with email and OTP verification.
- Users can log in with email and password.
- Verified users get full access after profile completion.

User Flows:
- User signs up, verifies OTP, creates password, and lands on the dashboard.
- User logs in and can complete profile setup manually or through an external verification provider.

Business Rules:
- Password must be 8 to 64 characters and include uppercase, lowercase, number, and special character.
- Account is locked after 5 failed login attempts.
- Profile setup is required before full feature access.

Edge Conditions:
- Invalid or expired OTP.
- Duplicate organization name.
- External verification provider failure.
- Missing mandatory profile fields.
"""


class GeneratorRefinementTests(unittest.TestCase):
    """Groups all integration and unit tests for the test generation and self-evaluation pipeline."""

    @patch("agents.test_generator.has_valid_openai_key", return_value=False)  # Simulate no API key so the fallback mode is used.
    def test_fallback_generation_runs_self_evaluation_loop(self, _mock_has_key):
        """
        Verifies that even in fallback mode (no LLM), generate_test_cases() still:
        1. Runs the full self-evaluation loop.
        2. Produces a report that contains all required sections.
        3. Generates at least 6 test cases.
        4. Resolves all coverage gaps so the final verdict is not 'GAPS_FOUND'.

        The underscore prefix on '_mock_has_key' signals that we don't need to use the mock
        object in the test body — we just need it active to force fallback mode.
        """
        report = generate_test_cases(SAMPLE_ANALYSIS, impact_analysis="")  # Run the pipeline in fallback mode.
        parsed_cases = parse_test_cases(report)  # Parse the returned markdown into structured objects for counting.

        self.assertIn("## Final Test Suite", report)            # The report must contain the main suite section header.
        self.assertIn("## Coverage Report", report)             # The report must contain the coverage evaluation section.
        self.assertIn("Coverage Score:", report)                 # A numeric coverage score must be present.
        self.assertIn("## Additional Test Cases Generated", report)  # Gap-filling tests must be listed in their own section.
        self.assertGreaterEqual(len(parsed_cases), 6)            # At least 6 distinct test cases must have been generated.
        self.assertNotIn("Coverage Verdict: GAPS_FOUND", report) # The loop must resolve all gaps before finishing.

    @patch("agents.test_generator.assign_test_case_ids")   # Mock ID assignment so we control the IDs ourselves in the test data.
    @patch("agents.test_generator._generate_gap_cases")    # Mock gap case generation to return a predictable additional test case.
    @patch("agents.test_generator._evaluate_coverage")     # Mock coverage evaluation to control what 'gaps' the loop sees.
    @patch("agents.test_generator._generate_initial_cases") # Mock initial generation to skip the real LLM call.
    def test_generation_does_not_stop_just_because_score_reaches_85(
        self,
        mock_generate_initial_cases,  # Controls what Pass 1 returns.
        mock_evaluate_coverage,       # Controls what coverage reports the evaluator returns.
        mock_generate_gap_cases,      # Controls what gap-filling cases the second pass returns.
        _mock_assign_ids,             # Suppresses ID assignment so our test IDs are preserved as-is.
    ):
        """
        Verifies that the refinement loop does NOT stop at a high coverage score like 85–90%.
        The only valid termination condition is 'FULL_COVERAGE' (all categories covered).
        This test sets up two evaluation cycles:
        - Cycle 1: score=90%, verdict='ACCEPTED_THRESHOLD_REACHED' → loop must continue.
        - Cycle 2: score=100%, verdict='FULL_COVERAGE' → loop must stop.
        """
        # Define a minimal 'initial' test case that Pass 1 returns.
        initial_case = TestCase(
            case_id="HP-001",
            title="Initial happy path",
            steps=["Open the workflow", "Complete the action"],
            expected_result="The action succeeds.",
            test_type="Happy Path",
            category="Functional and Non-functional",
            source_pass="Pass 1",
            coverage_dimensions=["Happy Path"],
        )
        # Define the additional test case that the gap-filling pass (Pass 2) returns.
        additional_case = TestCase(
            case_id="NEG-001",
            title="Gap-filling negative case",
            steps=["Trigger invalid input"],
            expected_result="The action is rejected.",
            test_type="Negative Scenarios",
            category="Positive, negative, and business-rule validation",
            source_pass="Pass 2 - Iteration 1",
            coverage_dimensions=["Negative Scenarios"],
        )

        # First coverage report: high score but NOT FULL_COVERAGE — loop must continue.
        report_85 = CoverageReport(
            categories=[
                CoverageCategoryReport("Happy Path", "covered", "covered", ["HP-001"]),
                CoverageCategoryReport("Negative Scenarios", "partial", "partial", []),  # Partial = gap still exists.
            ],
            coverage_score=90,
            overall_status="ACCEPTED_THRESHOLD_REACHED",  # Non-standard status — loop must treat it as GAPS_FOUND.
        )
        # Second coverage report: all covered — loop should stop after this.
        report_full = CoverageReport(
            categories=[
                CoverageCategoryReport("Happy Path", "covered", "covered", ["HP-001"]),
                CoverageCategoryReport("Negative Scenarios", "covered", "covered", ["NEG-001"]),  # Gap filled.
            ],
            coverage_score=100,
            overall_status="FULL_COVERAGE",  # This is the only status that terminates the loop.
        )

        # Configure mocks to return our controlled values.
        mock_generate_initial_cases.return_value = [initial_case]           # Pass 1 returns just the initial case.
        mock_evaluate_coverage.side_effect = [report_85, report_full]       # Evaluator returns report_85 first, then report_full.
        mock_generate_gap_cases.return_value = [additional_case]            # Gap fill returns the additional case.

        report = generate_test_cases(SAMPLE_ANALYSIS, impact_analysis="")  # Run the full pipeline with mocked internals.

        # The gap-fill function must have been called exactly once (for iteration 1 when gaps were found).
        self.assertEqual(mock_generate_gap_cases.call_count, 1)
        # The additional case's ID must appear in the final report.
        self.assertIn("NEG-001", report)
        # The final verdict must be FULL_COVERAGE since the loop terminated on report_full.
        self.assertIn("Coverage Verdict: FULL_COVERAGE", report)

    def test_initial_generation_categories_match_fixed_output_taxonomy(self):
        """
        Verifies that INITIAL_GENERATION_CATEGORIES (the list of LLM prompting slices for Pass 1)
        exactly matches OUTPUT_TEST_CATEGORIES (the fixed output category taxonomy).
        This is important because test cases generated for a given slice must be assignable
        to the corresponding output category without remapping.
        If these lists diverge, some test cases may never be correctly categorized.
        """
        self.assertEqual(INITIAL_GENERATION_CATEGORIES, OUTPUT_TEST_CATEGORIES)  # Both lists must be identical.

    def test_decorate_test_cases_maps_ad_hoc_categories_back_to_fixed_taxonomy(self):
        """
        Verifies that _decorate_test_cases() correctly maps an ad-hoc category label
        (like 'Coverage Gap Fill') to the appropriate fixed output category
        (like 'API contract, CRUD, integration, and service behavior') based on the test
        case's title, steps, expected result, and coverage dimensions.

        This ensures that even when the LLM or gap-fill logic assigns an unofficial category,
        the final output is always bucketed into one of the canonical output categories.
        """
        decorated = _decorate_test_cases(
            [
                TestCase(
                    title="Verify unauthorized API access is blocked",  # Title signals this is an API security test.
                    steps=["Call the protected API without valid auth."],  # Step confirms it's an API test.
                    expected_result="The API returns the correct auth failure and no state changes.",
                    test_type="Negative Scenarios",
                    category="Coverage Gap Fill",  # Ad-hoc category that needs remapping.
                    coverage_dimensions=["Negative Scenarios", "Integration & Dependency Failures"],  # Dimensions hint at API category.
                )
            ],
            "Functional and Non-functional",  # The fallback category to use if no better match is found.
            "Pass 2 - Iteration 1",           # The source pass label to tag onto the test case.
        )

        self.assertEqual(len(decorated), 1)  # One test case in, one decorated test case out.
        # The ad-hoc 'Coverage Gap Fill' category must have been mapped to the API-specific canonical category.
        self.assertEqual(
            decorated[0].category,
            "API contract, CRUD, integration, and service behavior",
        )

    def test_heuristic_coverage_requires_api_correctness_evidence(self):
        """
        Verifies that the heuristic coverage evaluator correctly identifies that having only
        integration/dependency tests is NOT sufficient for the 'Integration & Dependency Failures'
        coverage category — it also requires API correctness coverage (success codes, contract checks, etc.).

        This prevents the system from falsely marking a coverage area as 'covered' just because
        a few failure-path integration tests exist, when the core API contract is still untested.
        """
        # Build a report with 3 integration tests but NO API correctness or health-check tests.
        report = _build_heuristic_coverage_report(
            SAMPLE_ANALYSIS,
            "",  # No impact analysis — regression coverage is not applicable here.
            [
                TestCase(
                    case_id="INT-001",
                    title="Verify graceful timeout handling for dependency calls",
                    steps=["Trigger the workflow while a downstream dependency times out."],
                    expected_result="The system surfaces a recoverable timeout error.",
                    test_type="Integration & Dependency Failures",
                    category="Integration, billing, payment, third-party dependency, and failure-recovery scenarios",
                    coverage_dimensions=["Integration & Dependency Failures"],
                ),
                TestCase(
                    case_id="INT-002",
                    title="Verify retry behavior after partial third-party failure",
                    steps=["Cause one dependency to fail and retry the workflow."],
                    expected_result="The system retries safely and preserves data integrity.",
                    test_type="Integration & Dependency Failures",
                    category="Integration, billing, payment, third-party dependency, and failure-recovery scenarios",
                    coverage_dimensions=["Integration & Dependency Failures"],
                ),
                TestCase(
                    case_id="INT-003",
                    title="Verify rollback after downstream service rejection",
                    steps=["Trigger a downstream failure after partial completion."],
                    expected_result="The system rolls back safely without mixed states.",
                    test_type="Integration & Dependency Failures",
                    category="Integration, billing, payment, third-party dependency, and failure-recovery scenarios",
                    coverage_dimensions=["Integration & Dependency Failures"],
                ),
            ],
        )

        # Find the specific category report for 'Integration & Dependency Failures'.
        integration_report = next(
            item for item in report.categories if item.category == "Integration & Dependency Failures"
        )
        # The status must be 'partial' because API correctness coverage is missing.
        self.assertEqual(integration_report.status, "partial")
        # The missing_scenarios list must explicitly call out the need for API correctness coverage.
        self.assertTrue(
            any("API correctness coverage" in scenario for scenario in integration_report.missing_scenarios)
        )

    def test_heuristic_coverage_requires_true_security_coverage(self):
        """
        Verifies that the heuristic evaluator correctly marks 'Non-Functional Testing' as 'partial'
        even when the test suite has load, SLA, and endurance tests — because true security testing
        (auth bypass, data leakage, role-based access) is still missing.

        This quality gate prevents the system from declaring non-functional coverage 'complete'
        just because performance tests exist, when the security surface is completely untested.
        """
        # Build a report with only performance and endurance tests — no security tests.
        report = _build_heuristic_coverage_report(
            SAMPLE_ANALYSIS,
            "",  # No impact analysis.
            [
                TestCase(
                    case_id="NFT-001",
                    title="Verify load stability for the primary workflow",
                    steps=["Run the main workflow under concurrent load."],
                    expected_result="The workflow remains available under load.",
                    test_type="Non-Functional Testing",
                    category="System-wide performance, security, reliability, scalability, endurance, and compatibility",
                    coverage_dimensions=["Non-Functional Testing"],
                ),
                TestCase(
                    case_id="NFT-002",
                    title="Verify dashboard response time SLA",
                    steps=["Measure dashboard response time under normal operating conditions."],
                    expected_result="The dashboard meets the response-time SLA.",
                    test_type="Non-Functional Testing",
                    category="System-wide performance, security, reliability, scalability, endurance, and compatibility",
                    coverage_dimensions=["Non-Functional Testing"],
                ),
                TestCase(
                    case_id="NFT-003",
                    title="Verify long-running endurance across supported browsers",
                    steps=["Sustain realistic traffic for a long-running session and repeat it across browsers."],
                    expected_result="The platform stays stable and compatible across environments.",
                    test_type="Non-Functional Testing",
                    category="System-wide performance, security, reliability, scalability, endurance, and compatibility",
                    coverage_dimensions=["Non-Functional Testing"],
                ),
            ],
        )

        # Find the specific category report for 'Non-Functional Testing'.
        non_functional_report = next(
            item for item in report.categories if item.category == "Non-Functional Testing"
        )
        # Must be 'partial' because security tests are missing.
        self.assertEqual(non_functional_report.status, "partial")
        # The missing scenario must explicitly reference 'true security testing'.
        self.assertTrue(
            any("true security testing" in scenario.lower() for scenario in non_functional_report.missing_scenarios)
        )


if __name__ == "__main__":
    # Allows this test file to be run directly with 'python test_generator_refinement.py'.
    unittest.main()
