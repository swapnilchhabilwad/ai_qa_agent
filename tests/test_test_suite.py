import unittest  # Python's built-in testing framework.

# Import the core functions and classes from test_suite.py that we are testing.
from utils.test_suite import (
    CoverageCategoryReport,    # The dataclass representing coverage status for a single category.
    calculate_coverage_score,  # Function that computes the 0–100 coverage score from a list of category reports.
    deduplicate_test_cases,    # Function that removes test cases with identical content fingerprints.
    parse_test_cases,          # Function that parses raw markdown into structured TestCase objects.
)


# ─────────────────────────────────────────────────────────────────────
# SAMPLE REPORT — shared input fixture used across multiple test methods.
# Contains a minimal test suite and partial coverage report in the
# exact markdown format produced by the generate_test_cases() pipeline.
# ─────────────────────────────────────────────────────────────────────
SAMPLE_REPORT = """
## Final Test Suite

### Category: Functional and Non-functional

Test Case ID: HP-001
Title: Verify successful login
Preconditions:
- The user account exists.
Steps:
1. Open the login page.
2. Enter valid credentials.
3. Submit the form.
Expected Result: The user is redirected to the dashboard.
Test Type: Happy Path
Generation Source: Pass 1
Coverage Dimensions: Happy Path, UI/UX Validation

Test Case ID: NEG-001
Title: Verify invalid login is rejected
Steps:
1. Open the login page.
2. Enter invalid credentials.
3. Submit the form.
Expected Result: The system blocks login and shows an error.
Test Type: Negative Scenarios
Generation Source: Pass 1
Coverage Dimensions: Negative Scenarios

## Coverage Report
Coverage Score: 90%
Happy Path: ✅ Covered
"""


class TestSuiteParsingTests(unittest.TestCase):
    """Groups all unit tests for parse_test_cases(), deduplicate_test_cases(), and calculate_coverage_score()."""

    def test_parse_test_cases_extracts_structured_fields(self):
        """
        Verifies that parse_test_cases() correctly extracts all fields from the markdown report.
        This is the most important correctness test for the parser since it validates that:
        - The correct number of cases are extracted.
        - The case_id, category, preconditions, and step fields are parsed correctly.
        - Coverage dimension labels are extracted and normalized properly.
        """
        parsed_cases = parse_test_cases(SAMPLE_REPORT)  # Run the parser on the sample markdown.

        self.assertEqual(len(parsed_cases), 2)  # Exactly 2 test cases should be parsed from SAMPLE_REPORT.
        self.assertEqual(parsed_cases[0].case_id, "HP-001")  # The first test case ID must match.
        self.assertEqual(parsed_cases[0].category, "Functional and Non-functional")  # Category must be inherited from ### Category header.
        self.assertEqual(parsed_cases[0].preconditions, ["The user account exists."])  # Preconditions list must contain exactly one item.
        self.assertEqual(parsed_cases[0].steps[0], "Open the login page.")  # The first step must be correctly extracted.
        self.assertEqual(parsed_cases[0].coverage_dimensions, ["Happy Path", "UI/UX Validation"])  # Both dimensions must be parsed.
        self.assertEqual(parsed_cases[1].coverage_dimensions, ["Negative Scenarios"])  # Second case has a single dimension.

    def test_deduplicate_test_cases_removes_content_duplicates(self):
        """
        Verifies that deduplicate_test_cases() identifies and removes duplicate test cases
        even when the same case appears multiple times (as can happen across generation passes).
        Uses content fingerprinting (SHA-256 hash) rather than ID comparison.
        """
        parsed_cases = parse_test_cases(SAMPLE_REPORT)  # Parse the sample to get 2 unique test cases.
        duplicated = parsed_cases + [parsed_cases[0]]  # Manually duplicate the first test case to simulate a pass overlap.

        deduped_cases = deduplicate_test_cases(duplicated)  # Remove the duplicate.

        self.assertEqual(len(deduped_cases), 2)  # Only 2 unique cases should remain after deduplication.

    def test_coverage_score_uses_partial_credit(self):
        """
        Verifies the coverage score formula:
        - 'covered' = 1.0 point
        - 'partial' = 0.6 point
        - 'missing' = 0.0 point
        Score = (sum of points / total categories) * 100, rounded to nearest integer.

        With 7 covered (7.0) + 2 partial (1.2) + 1 missing (0.0) = 8.2 total across 10 categories:
        Expected score = round((8.2 / 10) * 100) = round(82.0) = 82.
        """
        reports = [
            CoverageCategoryReport("Happy Path", "covered", "Covered"),                    # 1.0 point
            CoverageCategoryReport("Negative Scenarios", "covered", "Covered"),             # 1.0 point
            CoverageCategoryReport("Boundary & Edge Cases", "covered", "Covered"),         # 1.0 point
            CoverageCategoryReport("State-Based Testing", "covered", "Covered"),           # 1.0 point
            CoverageCategoryReport("User Behavior Scenarios", "covered", "Covered"),       # 1.0 point
            CoverageCategoryReport("Integration & Dependency Failures", "covered", "Covered"),  # 1.0 point
            CoverageCategoryReport("UI/UX Validation", "covered", "Covered"),             # 1.0 point
            CoverageCategoryReport("Non-Functional Testing", "partial", "Partial"),        # 0.6 point
            CoverageCategoryReport("Data Persistence & Consistency", "partial", "Partial"),  # 0.6 point
            CoverageCategoryReport("Regression Coverage", "missing", "Missing"),           # 0.0 point
        ]

        self.assertEqual(calculate_coverage_score(reports), 82)  # Expected: round(8.2 / 10 * 100) = 82.


if __name__ == "__main__":
    # Allows this test file to be run directly with 'python test_test_suite.py'.
    unittest.main()
