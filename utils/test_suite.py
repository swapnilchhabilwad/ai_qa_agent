from __future__ import annotations  # Allows using class names as type hints before they are fully defined (e.g., TestCase | None).

import hashlib  # Used by TestCase.fingerprint() to generate a unique SHA-256 hash of a test case's content.
import re       # Used throughout for text normalization, cleaning bullet-point prefixes, and parsing markdown.
from dataclasses import dataclass, field  # 'dataclass' auto-generates __init__, 'field' provides mutable default values safely.
from typing import Iterable  # Allows functions to accept any iterable (list, generator, etc.) of TestCase objects.


# ─────────────────────────────────────────────────────────────────────
# COVERAGE TAXONOMY DEFINITION
# This is the single source of truth for all coverage categories.
# Every generated test case is assigned to one of these 20 buckets.
# The order here determines the order categories appear in the final report.
# ─────────────────────────────────────────────────────────────────────
COVERAGE_CATEGORIES = [
    "Foundational Smoke and Sanity",               # Minimal smoke checks and basic system reachability.
    "End-to-End Business Workflows",               # Full user journeys across the system.
    "Core Functional Requirements",                # Core feature behaviors from the PRD.
    "Detailed Functional Rules and Constraints",   # Business rules, entitlements, role-aware logic.
    "Non-Functional Expectations",                 # Reliability, availability, robustness.
    "Front-end UI and UX",                         # UI rendering, labels, buttons, usability.
    "Front-end Responsive and Cross-Browser",      # Mobile, tablet, and cross-browser tests.
    "Back-end Database Persistence and Integrity", # Data saving, stored values, integrity.
    "Back-end Schema and Audit Validation",        # DB schema correctness and audit trails.
    "API Contract and Schema Validation",          # API contract, status codes, JSON schema.
    "API CRUD and State Change Correctness",       # API-driven create/update/delete operations.
    "Positive Happy Path Scenarios",               # Clean success flows with valid inputs.
    "Negative Error Handling and Validation Scenarios",  # Invalid inputs, rejections, error messages.
    "State Transitions and Lifecycle flows",       # Lifecycle state changes (e.g., active → completed).
    "User Behavior",                               # Refresh, back navigation, multi-tab patterns.
    "Concurrency and Race Conditions",             # Repeated/parallel actions creating race conditions.
    "Interruption and Recovery scenarios",         # Timeouts, disconnections, and retry/recovery.
    "Boundary and Edge Cases",                     # Min/max limits, empty/null inputs, thresholds.
    "System Performance and Latency SLA",          # Load, response-time SLA, scalability.
    "Security, Auth, and Role-Based Access Control",  # Authentication, authorization, data leakage.
]

# Maps coverage status strings to numeric scores used in calculate_coverage_score().
# 'covered' = full credit, 'partial' = partial credit, 'missing' = no credit.
STATUS_POINTS = {
    "covered": 1.0,   # This category has been thoroughly tested.
    "partial": 0.6,   # Some coverage exists but it is not comprehensive.
    "missing": 0.0,   # No test cases target this category at all.
}

# Maps coverage status strings to emoji icons for use in the human-readable coverage report.
STATUS_ICONS = {
    "covered": "✅",  # Green checkmark = fully covered.
    "partial": "⚠️",  # Warning sign = only partially covered.
    "missing": "❌",  # Red X = not covered at all.
}

# Maps each coverage category to its short prefix string.
# These prefixes are used when generating unique test case IDs (e.g., "PHS-001" for Positive Happy Path).
CATEGORY_PREFIXES = {
    "Foundational Smoke and Sanity": "FSS",
    "End-to-End Business Workflows": "EBW",
    "Core Functional Requirements": "CFR",
    "Detailed Functional Rules and Constraints": "DFR",
    "Non-Functional Expectations": "NFE",
    "Front-end UI and UX": "FUI",
    "Front-end Responsive and Cross-Browser": "FRC",
    "Back-end Database Persistence and Integrity": "BPI",
    "Back-end Schema and Audit Validation": "BSA",
    "API Contract and Schema Validation": "ACS",
    "API CRUD and State Change Correctness": "ACC",
    "Positive Happy Path Scenarios": "PHS",
    "Negative Error Handling and Validation Scenarios": "NEH",
    "State Transitions and Lifecycle flows": "STL",
    "User Behavior": "USB",
    "Concurrency and Race Conditions": "CRC",
    "Interruption and Recovery scenarios": "IRS",
    "Boundary and Edge Cases": "BEC",
    "System Performance and Latency SLA": "SPL",
    "Security, Auth, and Role-Based Access Control": "SAR",
}

# Maps each coverage category to a set of keyword hints.
# These keywords are used to heuristically infer which category a test case belongs to
# when the category field is missing or ambiguous.
CATEGORY_KEYWORDS = {
    "Foundational Smoke and Sanity": ("smoke", "sanity", "health", "basic navigation", "minimal working system", "heartbeat", "availability"),
    "End-to-End Business Workflows": ("end-to-end", "e2e", "workflow", "journey", "business process", "complete flow"),
    "Core Functional Requirements": ("functional", "core requirement", "primary feature", "main functionality"),
    "Detailed Functional Rules and Constraints": ("rule", "constraint", "business rule", "entitlement", "eligibility", "role-aware", "permission"),
    "Non-Functional Expectations": ("reliability", "availability", "error resilience", "robustness", "non-functional"),
    "Front-end UI and UX": ("ui", "ux", "layout", "label", "button", "copy", "usability", "frontend", "interface"),
    "Front-end Responsive and Cross-Browser": ("responsive", "mobile", "tablet", "viewport", "browser", "chrome", "safari", "firefox", "edge"),
    "Back-end Database Persistence and Integrity": ("database", "persistence", "save", "stored", "integrity", "data persistence", "backend record"),
    "Back-end Schema and Audit Validation": ("schema", "audit", "default value", "record structure", "db-level", "audit trail"),
    "API Contract and Schema Validation": ("api contract", "api schema", "status code", "200", "201", "json schema"),
    "API CRUD and State Change Correctness": ("api crud", "api create", "api update", "api delete", "state change", "service layer"),
    "Positive Happy Path Scenarios": ("happy path", "success", "successful", "positive", "valid input"),
    "Negative Error Handling and Validation Scenarios": ("negative", "invalid", "error", "reject", "denied", "blocked", "failure", "validation error"),
    "State Transitions and Lifecycle flows": ("state", "transition", "lifecycle", "status", "active", "inactive", "pending"),
    "User Behavior": ("refresh", "back navigation", "multi-tab", "multi tab", "page reload", "browser back"),
    "Concurrency and Race Conditions": ("concurrency", "race condition", "repeated action", "simultaneous", "parallel"),
    "Interruption and Recovery scenarios": ("interruption", "recovery", "timeout", "disconnected", "retry flow"),
    "Boundary and Edge Cases": ("boundary", "edge", "limit", "threshold", "minimum", "maximum", "empty", "null"),
    "System Performance and Latency SLA": ("performance", "latency", "sla", "load", "stress", "scalability", "response time"),
    "Security, Auth, and Role-Based Access Control": ("security", "auth", "rbac", "authentication", "authorization", "bypass", "data leakage", "unauthorized"),
}


# ─────────────────────────────────────────────────────────────────────
# DATA MODELS
# These dataclasses are the shared data contract used throughout the
# entire framework — from generation to evaluation to CSV export.
# ─────────────────────────────────────────────────────────────────────

@dataclass  # Automatically generates __init__, __repr__, and __eq__ methods.
class TestCase:
    """
    Represents a single, fully-structured test case.
    All test cases generated by the AI or fallback are stored as TestCase instances.
    """
    case_id: str = ""                                   # Unique identifier, e.g., "PHS-001". Assigned after generation.
    title: str = ""                                     # Short description of what the test validates.
    preconditions: list[str] = field(default_factory=list)   # Setup steps required before the test begins.
    steps: list[str] = field(default_factory=list)           # The exact actions to perform during the test.
    expected_result: str = ""                           # The outcome that confirms the test passed.
    test_type: str = ""                                 # Testing dimension label (e.g., "Happy Path", "Security").
    category: str = "General"                           # Coverage bucket this test belongs to (from COVERAGE_CATEGORIES).
    source_pass: str = "Pass 1"                         # Which generation pass produced this case.
    coverage_dimensions: list[str] = field(default_factory=list)  # Labels that link this case to coverage categories.

    def fingerprint(self) -> str:
        """
        Generates a SHA-256 content hash unique to this test case's key fields.
        Used by deduplicate_test_cases() to identify and remove duplicate cases,
        even if they were generated in different passes with different IDs.
        """
        # Normalize and join the core content fields so minor whitespace/casing differences don't create false uniqueness.
        parts = [
            normalize_text(self.title),                         # Normalize the test title.
            normalize_text(" ".join(self.preconditions)),       # Normalize all preconditions as one block.
            normalize_text(" ".join(self.steps)),               # Normalize all steps as one block.
            normalize_text(self.expected_result),               # Normalize the expected result.
            normalize_text(self.test_type),                     # Include the test type to differentiate otherwise identical cases.
        ]
        # Join parts with a pipe separator and hash the result. The pipe prevents "a b" + "c" from colliding with "a" + "b c".
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass
class CoverageCategoryReport:
    """
    Represents the coverage evaluation result for a single category (e.g., 'Happy Path').
    This is generated by the AI evaluator or the heuristic coverage report builder.
    """
    category: str                                          # The name of the coverage category being assessed.
    status: str                                            # One of: 'covered', 'partial', 'missing'.
    reason: str                                            # Plain-text explanation of why this status was assigned.
    evidence_ids: list[str] = field(default_factory=list)          # IDs of test cases that contribute to this category.
    missing_scenarios: list[str] = field(default_factory=list)     # Specific testing scenarios that are still absent.
    untested_risks: list[str] = field(default_factory=list)        # What bugs or regressions could slip through if not covered.


@dataclass
class CoverageReport:
    """
    The aggregate result of evaluating an entire test suite's coverage.
    Combines per-category data with global metrics for the final report.
    """
    categories: list[CoverageCategoryReport]              # One report entry per coverage category.
    missing_scenarios: list[str] = field(default_factory=list)   # Global list of missing scenarios across all categories.
    untested_risks: list[str] = field(default_factory=list)      # Global list of risks that remain untested.
    weak_coverage_areas: list[str] = field(default_factory=list) # Human-readable strings describing where coverage is thin.
    coverage_score: int = 0                               # Numeric score (0–100) derived from the status of all categories.
    overall_status: str = "GAPS_FOUND"                    # Either 'FULL_COVERAGE' or 'GAPS_FOUND'.
    iterations_used: int = 1                              # How many refinement iterations the pipeline ran.


@dataclass
class TestSuiteResult:
    """
    The complete output bundle returned by generate_test_cases().
    Contains the final test suite, coverage report, and metadata about the generation process.
    """
    final_test_suite: list[TestCase]                      # All generated test cases (deduplicated and sorted).
    coverage_report: CoverageReport                       # The final coverage evaluation across all categories.
    initial_gaps: list[str] = field(default_factory=list)          # The gaps identified after Pass 1 before any refinement.
    additional_test_cases: list[TestCase] = field(default_factory=list)  # Cases added during Pass 2 refinement iterations.


# ─────────────────────────────────────────────────────────────────────
# SHARED TEXT UTILITY FUNCTIONS
# These pure helper functions are used across many parts of the system
# for consistent text processing and comparison.
# ─────────────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    """
    Collapses multiple whitespace characters into a single space and converts to lowercase.
    Used to ensure that minor formatting differences don't prevent string comparisons from matching.
    """
    return re.sub(r"\s+", " ", value or "").strip().lower()  # 'value or ""' handles None safely without crashing.


def clean_list_item(value: str) -> str:
    """
    Strips common list-item prefixes (hyphens, asterisks, numbered list markers) from a string.
    Used when parsing markdown bullet points or numbered steps from the LLM output.
    Example: '- Click the button.' → 'Click the button.'
    """
    # Remove leading whitespace, then optional markdown bullets (-, *, 1., 2)) at the start.
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value or "").strip()
    return cleaned.strip("* ").strip()  # Also strip stray asterisks used for emphasis.


def keyword_matches_text(text: str, keyword: str) -> bool:
    """
    Performs a whole-word keyword search using regex.
    Unlike a simple 'in' check, this ensures 'api' won't match inside 'capability'.
    The (?<!\w) and (?!\w) are zero-width lookbehind/lookahead assertions to enforce word boundaries.
    """
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


def normalize_dimension_name(value: str) -> str:
    """
    Attempts to map an arbitrary coverage dimension label back to the official COVERAGE_CATEGORIES name.
    This handles cases where the LLM uses a slightly different phrasing for the same category.
    First tries an exact match, then falls back to keyword matching.
    """
    normalized = normalize_text(value)  # Convert to lowercase for comparison.

    # Try an exact case-insensitive match against the official category list.
    for category in CATEGORY_KEYWORDS:
        if normalized == normalize_text(category):
            return category  # Exact match found — return the canonical name.

    # If no exact match, try keyword-based matching (e.g., 'happy path' maps to 'Positive Happy Path Scenarios').
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword_matches_text(normalized, keyword) for keyword in keywords):
            return category  # Keyword match found — return the canonical name.

    return value.strip()  # If no match is found at all, return the original value as-is.


def infer_primary_dimension(test_case: TestCase) -> str:
    """
    Infers the most relevant coverage dimension for a test case by scanning its fields.
    Used as a fallback when coverage_dimensions is empty or contains non-standard labels.
    Priority: explicit dimensions → test_type → category → title → steps → expected_result.
    """
    # First, try to find a recognized dimension in the explicitly assigned coverage_dimensions list.
    for value in test_case.coverage_dimensions:
        normalized = normalize_dimension_name(value)
        if normalized in CATEGORY_PREFIXES:  # Only accept officially recognized categories.
            return normalized

    # Build a combined text block from all informative fields for keyword scanning.
    haystacks = [
        test_case.test_type,          # The declared test type is often the most direct signal.
        test_case.category,           # The assigned category bucket.
        test_case.title,              # The test title often names the scenario type.
        " ".join(test_case.steps),    # The steps may contain type-specific verbs (e.g., 'navigate', 'concurrent').
        test_case.expected_result,    # The expected result may reference specific outcomes.
    ]
    normalized = normalize_text(" ".join(haystacks))  # Merge all fields into one searchable string.

    # Scan for keyword matches against every known category.
    for category, keywords in CATEGORY_KEYWORDS.items():
        if normalize_text(category) in normalized or any(
            keyword_matches_text(normalized, keyword) for keyword in keywords
        ):
            return category  # Return the first category whose keywords appear in the combined text.

    return "Happy Path"  # Ultimate fallback — if nothing matches, assume it's a happy path test.


def calculate_coverage_score(category_reports: Iterable[CoverageCategoryReport]) -> int:
    """
    Computes a 0–100 coverage score based on the status of each category report.
    'covered' = 1.0 point, 'partial' = 0.6 points, 'missing' = 0.0 points.
    The score is the average of all category scores, scaled to a percentage.
    """
    reports = list(category_reports)  # Materialize the iterable once so we can count it.
    if not reports:
        return 0  # Avoid division by zero if there are no reports.

    # Sum the partial credit scores for each category.
    total = sum(STATUS_POINTS.get(report.status, 0.0) for report in reports)
    # Divide by the number of categories to get the average and scale to 0–100.
    return round((total / len(reports)) * 100)


def status_with_icon(status: str) -> str:
    """
    Converts a raw status string ('covered', 'partial', 'missing') to a human-readable
    label with an emoji prefix (e.g., '✅ Covered', '⚠️ Partially Covered', '❌ Missing').
    """
    # Map the raw status to a display label, falling back to title-casing for unknown values.
    label = {
        "covered": "Covered",
        "partial": "Partially Covered",
        "missing": "Missing",
    }.get(status, status.title())
    return f"{STATUS_ICONS.get(status, '')} {label}".strip()  # Combine the icon and label, removing any trailing whitespace.


def _determine_coverage_verdict(report: CoverageReport) -> str:
    """
    Checks whether every category in the report is marked 'covered'.
    Returns 'FULL_COVERAGE' if all are covered, otherwise 'GAPS_FOUND'.
    This verdict controls whether the iterative refinement loop terminates.
    """
    if all(item.status == "covered" for item in report.categories):
        return "FULL_COVERAGE"  # All categories are satisfied — no more generation needed.
    return "GAPS_FOUND"  # At least one category is still 'partial' or 'missing'.


# ─────────────────────────────────────────────────────────────────────
# MARKDOWN PARSER
# Converts the AI's raw markdown output into structured TestCase objects.
# ─────────────────────────────────────────────────────────────────────

def parse_test_cases(markdown_content: str) -> list[TestCase]:
    """
    State-machine based markdown parser.
    Reads the LLM's markdown output line by line and assembles TestCase objects
    by identifying headers, field labels, and list items.

    The expected markdown format per test case is:
        ### Category: <name>
        Test Case ID: <id>
        Title: <title>
        Preconditions:
        - <precondition>
        Steps:
        - <step>
        Expected Result: <result>
        Test Type: <type>
        Coverage Dimensions: <dim1>, <dim2>
    """
    current_category = "General"    # Tracks the active ### Category: header for grouping test cases.
    current_case: TestCase | None = None  # Holds the test case currently being assembled.
    current_field = ""              # Tracks which field (e.g., 'steps', 'preconditions') we're currently reading into.
    parsed_cases: list[TestCase] = []  # The final list of fully-parsed test cases.

    # Compile regex patterns for each field label. re.IGNORECASE handles formatting inconsistencies.
    field_patterns = {
        "title": re.compile(r"^\**Title\**:\s*(.*)$", re.IGNORECASE),
        "preconditions": re.compile(r"^\**Preconditions\**:\s*(.*)$", re.IGNORECASE),
        "steps": re.compile(r"^\**Steps\**:\s*(.*)$", re.IGNORECASE),
        "expected_result": re.compile(r"^\**(?:Expected Result|Expected Outcome)\**:\s*(.*)$", re.IGNORECASE),
        "test_type": re.compile(r"^\**Test Type\**:\s*(.*)$", re.IGNORECASE),
        "source_pass": re.compile(r"^\**(?:Generation Source|Source Pass|Origin)\**:\s*(.*)$", re.IGNORECASE),
        "coverage_dimensions": re.compile(r"^\**Coverage Dimensions\**:\s*(.*)$", re.IGNORECASE),
    }
    # Separate pattern for Test Case ID since it triggers starting a new TestCase object.
    id_pattern = re.compile(r"^\**(?:Test Case ID|ID)\**:\s*(.*)$", re.IGNORECASE)

    def flush_current_case():
        """
        Inner helper: finalizes and validates the current test case and appends it to the result list.
        Called whenever the parser encounters a new test case or section header.
        A case is only accepted if it has an ID, a title, and at least one step (minimum valid structure).
        """
        nonlocal current_case  # 'nonlocal' lets this inner function modify the outer variable.
        if not current_case:
            return  # Nothing to flush if no test case is in progress.

        # Strip whitespace from all string fields to prevent mismatches in downstream comparisons.
        current_case.title = current_case.title.strip()
        current_case.expected_result = current_case.expected_result.strip()
        current_case.test_type = current_case.test_type.strip()
        current_case.source_pass = current_case.source_pass.strip() or "Pass 1"  # Default to 'Pass 1' if missing.
        current_case.preconditions = [item for item in current_case.preconditions if item]   # Remove empty strings.
        current_case.steps = [item for item in current_case.steps if item]                   # Remove empty strings.
        current_case.coverage_dimensions = [item for item in current_case.coverage_dimensions if item]  # Remove empty strings.

        # Only accept the case if it has the minimum required fields to be a valid test case.
        if current_case.case_id and current_case.title and current_case.steps:
            parsed_cases.append(current_case)
        current_case = None  # Reset so the next case starts fresh.

    # Iterate through each line of the markdown content.
    for raw_line in markdown_content.splitlines():
        line = raw_line.rstrip()       # Remove trailing whitespace (but preserve leading indentation).
        line_strip = line.strip()      # Fully stripped version used for pattern matching.

        # Detect section headers like '## Final Test Suite' and reset state.
        if line_strip.startswith("## "):
            flush_current_case()       # Save the current case before moving to a new section.
            current_field = ""         # Reset field tracking since we're between sections.
            continue

        # Detect category headers like '### Category: Positive Happy Path Scenarios'.
        if line_strip.startswith("### Category:"):
            flush_current_case()       # Save the current case before processing the new category.
            # Extract the category name after the prefix, defaulting to 'General' if empty.
            current_category = line_strip.replace("### Category:", "", 1).strip() or "General"
            current_field = ""         # Reset field tracking since we just finished a category block.
            continue

        # Detect a Test Case ID line — this marks the beginning of a new test case.
        id_match = id_pattern.match(line_strip)
        if id_match:
            flush_current_case()       # Finalize any previous test case before starting this one.
            # Create a new TestCase and inherit the current category from the last '### Category:' header.
            current_case = TestCase(
                case_id=id_match.group(1).strip().strip("*# "),  # Extract the ID value, removing markdown asterisks.
                category=current_category,
            )
            current_field = ""         # Reset field tracking for the new case.
            continue

        # If no test case is currently being assembled, skip this line.
        if not current_case:
            continue

        # Try to match the current line against each known field label pattern.
        matched_new_field = False
        for field_name, pattern in field_patterns.items():
            field_match = pattern.match(line_strip)
            if not field_match:
                continue  # This field pattern didn't match — try the next one.

            matched_new_field = True   # We found the field this line belongs to.
            current_field = field_name # Update the active field so continuation lines go to the right place.
            content = field_match.group(1).strip().strip("* ")  # Extract the value on the same line as the label.

            # Assign the extracted value to the correct field on the current TestCase.
            if field_name == "title":
                current_case.title = content  # Titles are single-line strings.
            elif field_name == "preconditions":
                if content:  # Only append if there is inline content (not just the label line).
                    current_case.preconditions.append(clean_list_item(content))
            elif field_name == "steps":
                if content:  # Only append if there is inline content.
                    current_case.steps.append(clean_list_item(content))
            elif field_name == "expected_result":
                current_case.expected_result = content  # Expected result is a single string.
            elif field_name == "test_type":
                current_case.test_type = content  # Test type is a single string.
            elif field_name == "source_pass":
                current_case.source_pass = content  # Source pass is a single string.
            elif field_name == "coverage_dimensions":
                # Coverage dimensions are comma or semicolon separated on one line — split and normalize each.
                current_case.coverage_dimensions.extend(
                    normalize_dimension_name(item)
                    for item in re.split(r",|;", content)
                    if item.strip()
                )
            break  # Stop checking patterns once we've found a match for this line.

        if matched_new_field:
            continue  # This line was a field label — move to the next line.

        if not line_strip:
            continue  # Skip blank lines between fields.

        # This line is a continuation of the previous field (e.g., a bullet point below 'Steps:').
        # Route it to the correct field based on what we last encountered.
        if current_field == "title":
            current_case.title = f"{current_case.title} {line_strip}".strip()  # Append wrapped title text.
        elif current_field == "preconditions":
            current_case.preconditions.append(clean_list_item(line_strip))  # Append a new precondition bullet.
        elif current_field == "steps":
            current_case.steps.append(clean_list_item(line_strip))  # Append a new step bullet.
        elif current_field == "expected_result":
            current_case.expected_result = f"{current_case.expected_result} {line_strip}".strip()  # Append wrapped result text.
        elif current_field == "test_type":
            current_case.test_type = f"{current_case.test_type} {line_strip}".strip()  # Append wrapped test type.
        elif current_field == "source_pass":
            current_case.source_pass = f"{current_case.source_pass} {line_strip}".strip()  # Append wrapped source pass.
        elif current_field == "coverage_dimensions":
            # Continuation lines for coverage dimensions — parse them the same way as the first line.
            current_case.coverage_dimensions.extend(
                normalize_dimension_name(item)
                for item in re.split(r",|;", line_strip)
                if item.strip()
            )

    flush_current_case()  # Finalize the last test case in the file after the loop ends.
    return parsed_cases   # Return the complete list of structured TestCase objects.


def deduplicate_test_cases(test_cases: Iterable[TestCase]) -> list[TestCase]:
    """
    Removes test cases with identical content fingerprints from the list.
    The fingerprint is a SHA-256 hash of the normalized title, steps, expected result, and type,
    so semantically identical tests are removed even if they have different IDs or were generated
    in different passes.
    """
    deduplicated: list[TestCase] = []    # Output list preserving the first occurrence of each unique case.
    seen_fingerprints: set[str] = set()  # Fast set-based lookup to detect duplicates in O(1) time.

    for test_case in test_cases:
        fingerprint = test_case.fingerprint()  # Compute the content-based hash.
        if fingerprint in seen_fingerprints:
            continue  # Skip this case — an identical one was already added.
        seen_fingerprints.add(fingerprint)   # Register this fingerprint as seen.
        deduplicated.append(test_case)       # Keep the first occurrence.

    return deduplicated


def assign_test_case_ids(test_cases: list[TestCase], existing_cases: Iterable[TestCase] | None = None):
    """
    Assigns or re-assigns sequential IDs to test cases.
    IDs are formatted as '<PREFIX>-<NNN>' where:
    - PREFIX comes from CATEGORY_PREFIXES based on the test's primary coverage dimension.
    - NNN is a zero-padded 3-digit number (e.g., '001', '042').
    Cases that already have a valid, non-conflicting, correctly-prefixed ID are left unchanged.
    This function modifies the test_cases list in-place.
    """
    existing = list(existing_cases or [])  # Materialize existing cases (could be a generator).
    # Build a set of all already-used IDs (uppercased for case-insensitive comparison).
    used_ids = {case.case_id.upper() for case in existing if case.case_id}
    # Track the highest number used for each prefix so we can continue counting from there.
    counters: dict[str, int] = {}

    # Pre-scan existing cases to establish the current counter state for each prefix.
    for case in existing:
        if "-" not in case.case_id:
            continue  # Skip IDs without a dash (they don't follow the PREFIX-NNN format).
        prefix, _, suffix = case.case_id.partition("-")  # Split 'PHS-042' → ('PHS', '-', '042').
        if suffix.isdigit():
            # Ensure the counter starts after the highest already-used number for this prefix.
            counters[prefix] = max(counters.get(prefix, 0), int(suffix))

    # Now assign IDs to the new test cases.
    for test_case in test_cases:
        # Determine the correct prefix based on the test case's primary coverage dimension.
        desired_prefix = CATEGORY_PREFIXES.get(infer_primary_dimension(test_case), "TC")
        current_id = test_case.case_id.upper()  # Uppercase for comparison.

        # Keep the existing ID if it is valid, unique, and already uses the correct prefix.
        if current_id and current_id not in used_ids and current_id.startswith(desired_prefix):
            used_ids.add(current_id)  # Mark it as used so no future case claims the same ID.
            # Update the counter so future IDs start after this one.
            if "-" in current_id:
                prefix, _, suffix = current_id.partition("-")
                if suffix.isdigit():
                    counters[prefix] = max(counters.get(prefix, 0), int(suffix))
            continue  # No change needed for this test case.

        # Generate the next available ID for this prefix.
        next_number = counters.get(desired_prefix, 0) + 1
        new_id = f"{desired_prefix}-{next_number:03d}"  # Format as 'PHS-001', 'PHS-042', etc.

        # Handle collisions by incrementing until a free ID is found.
        while new_id.upper() in used_ids:
            next_number += 1
            new_id = f"{desired_prefix}-{next_number:03d}"

        # Commit the new ID to the test case and register it as used.
        counters[desired_prefix] = next_number
        test_case.case_id = new_id
        used_ids.add(new_id.upper())


# ─────────────────────────────────────────────────────────────────────
# RENDERING FUNCTIONS
# Convert internal data structures into human-readable markdown strings
# for the final test suite report.
# ─────────────────────────────────────────────────────────────────────

def render_test_case(test_case: TestCase) -> str:
    """
    Converts a single TestCase object into the standard markdown block format.
    This is the inverse of the parse_test_cases() function.
    """
    lines = [
        f"Test Case ID: {test_case.case_id}",  # Unique identifier line.
        f"Title: {test_case.title}",            # Scenario description line.
    ]

    if test_case.preconditions:  # Only add the Preconditions section if there are any.
        lines.append("Preconditions:")
        lines.extend(f"- {item}" for item in test_case.preconditions)  # Add each precondition as a bullet point.

    lines.append("Steps:")
    # Add each step as a numbered list item (1., 2., 3., ...).
    lines.extend(f"{index}. {step}" for index, step in enumerate(test_case.steps, start=1))
    lines.append(f"Expected Result: {test_case.expected_result}")  # The pass/fail criterion.

    if test_case.test_type:  # Only add Test Type if it was assigned.
        lines.append(f"Test Type: {test_case.test_type}")

    if test_case.source_pass:  # Only add Generation Source if it was set.
        lines.append(f"Generation Source: {test_case.source_pass}")

    if test_case.coverage_dimensions:  # Only add Coverage Dimensions if the list is non-empty.
        lines.append(f"Coverage Dimensions: {', '.join(test_case.coverage_dimensions)}")

    return "\n".join(lines)  # Join all lines into a single multi-line string.


def render_test_suite(test_cases: Iterable[TestCase]) -> str:
    """
    Renders a full test suite as markdown, grouping test cases under their category headers.
    Each time the category changes, a new '### Category: ...' header is inserted.
    """
    blocks: list[str] = []       # Accumulates rendered blocks (headers and test cases).
    current_category = None      # Tracks the last category we wrote a header for.

    for test_case in test_cases:
        if test_case.category != current_category:  # If the category changed, write a new section header.
            current_category = test_case.category
            blocks.append(f"### Category: {current_category}")
        blocks.append(render_test_case(test_case))  # Append the formatted test case block.

    return "\n\n".join(blocks)  # Separate each block with a blank line for readable markdown output.


def render_coverage_report(report: CoverageReport) -> str:
    """
    Renders the coverage report as a human-readable markdown section.
    Includes the overall score, verdict, per-category detail, and global risk summaries.
    """
    lines = [
        "## Coverage Report",
        f"Coverage Score: {report.coverage_score}%",          # Overall numeric score (0–100).
        f"Coverage Verdict: {report.overall_status}",         # FULL_COVERAGE or GAPS_FOUND.
        f"Iterations Used: {report.iterations_used}",         # How many refinement loops ran.
        "",
    ]

    # Add a detailed section for each coverage category.
    for category_report in report.categories:
        lines.append(f"{category_report.category}: {status_with_icon(category_report.status)}")  # Category + icon.
        lines.append(f"Reason: {category_report.reason}")  # Explanation for the status.
        if category_report.evidence_ids:  # Only show evidence IDs if any exist.
            lines.append(f"Evidence IDs: {', '.join(category_report.evidence_ids)}")
        if category_report.missing_scenarios:  # Only show missing scenarios if any were identified.
            lines.append(f"Missing Scenarios: {'; '.join(category_report.missing_scenarios)}")
        if category_report.untested_risks:  # Only show untested risks if any were identified.
            lines.append(f"Untested Risks: {'; '.join(category_report.untested_risks)}")
        lines.append("")  # Blank line between category blocks for readability.

    if report.missing_scenarios:  # Only add global missing scenarios section if any exist.
        lines.append("Global Missing Scenarios:")
        lines.extend(f"- {item}" for item in report.missing_scenarios)
        lines.append("")

    if report.untested_risks:  # Only add global untested risks section if any exist.
        lines.append("Untested Risks:")
        lines.extend(f"- {item}" for item in report.untested_risks)
        lines.append("")

    if report.weak_coverage_areas:  # Only add weak coverage areas section if any exist.
        lines.append("Weak Coverage Areas:")
        lines.extend(f"- {item}" for item in report.weak_coverage_areas)

    return "\n".join(lines).strip()  # Join all lines and remove leading/trailing blank lines.


def render_final_report(result: TestSuiteResult) -> str:
    """
    The top-level rendering function called at the very end of generate_test_cases().
    Converts the complete TestSuiteResult into a single markdown string for display and export.
    Currently only renders the final test suite (not the full coverage report or gap details).
    """
    return render_test_suite(result.final_test_suite).strip()  # Render and trim whitespace from the full suite.
