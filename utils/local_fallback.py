import re  # Used for pattern matching to extract sentences and keywords from text.
from collections import Counter  # Used to find the most frequent words to 'infer' the subject of a PRD.


# Regex to find alphanumeric tokens (words) for keyword analysis.
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Regex to split a large block of text into individual sentences based on punctuation or newlines.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# List of common 'filler' words to ignore when trying to find the most important subject of a document.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "user", "users", "with", "will", "this", "their", "them", "into",
    "when", "where", "while", "must", "should", "can", "only",
}


def tokenize(text: str) -> list[str]:
    """Converts a string into a list of lowercase words, filtering out short/invalid tokens."""
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2]


def extract_sentences(text: str) -> list[str]:
    """
    Cleans and splits a text block into a list of unique, non-empty sentences.
    Used to create a 'pool' of information for the fallback analyzer.
    """
    sentences: list[str] = []

    # Split by the predefined regex (punctuation or newlines).
    for chunk in SENTENCE_RE.split(text):
        # Remove extra internal spaces and leading/trailing dashes/spaces.
        sentence = " ".join(chunk.split()).strip(" -")
        if sentence and sentence not in sentences:
            sentences.append(sentence)

    return sentences


def _select_sentences(
    sentences: list[str], keywords: set[str], fallback_pool: list[str], limit: int = 4
) -> list[str]:
    """
    Heuristic-based search:
    Finds sentences that contain specific keywords and ranks them by 'relevance' (keyword density).
    If not enough sentences are found, it pulls from the start of the document as a fallback.
    """
    ranked: list[tuple[int, int, str]] = []

    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        # Count how many target keywords appear in this sentence.
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        if score:
            # We use negative index to prioritize sentences found earlier in the document if scores are tied.
            ranked.append((score, -index, sentence))

    # Sort descending by score.
    ranked.sort(reverse=True)
    selected = [sentence for _, _, sentence in ranked[:limit]]

    # If the search was too strict, fill the remaining slots with general text from the document.
    if len(selected) < limit:
        for sentence in fallback_pool:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) == limit:
                break

    return selected[:limit]


def infer_subject(text: str) -> str:
    """
    Identifies what the document is about (e.g., 'login', 'checkout') by checking for 
    high-priority keywords or finding the most common non-stopword tokens.
    """
    lowered = text.lower()

    # Priority 1: Check for explicit common web app functional phrases.
    for phrase in (
        "login", "signup", "registration", "search", "checkout",
        "payment", "profile", "password reset", "authentication",
    ):
        if phrase in lowered:
            return phrase

    # Priority 2: Statistically analyze the text for the most frequent important words.
    counts = Counter(token for token in tokenize(text) if token not in STOPWORDS)
    common = [token for token, _ in counts.most_common(3)]

    if common:
        return " ".join(common)

    # Priority 3: Default generic string.
    return "the described workflow"


def build_analysis(context: str) -> str:
    """
    Acts as a 'Mock AI Analyzer'.
    It uses regex and keyword matching to simulate the structural output of an LLM.
    This allows the framework to function offline or without an API key.
    """
    sentences = extract_sentences(context)
    # Define a default set of sentences to use if no keyword matches are found.
    fallback_pool = sentences[:6] if sentences else ["No requirement details were extracted."]

    # Categorize sentences into Features, Flows, Rules, and Edges based on keyword sets.
    features = _select_sentences(
        sentences,
        {"allow", "support", "feature", "screen", "page", "login", "create", "view"},
        fallback_pool,
    )
    flows = _select_sentences(
        sentences,
        {"user", "enter", "click", "submit", "navigate", "select", "open"},
        fallback_pool,
    )
    rules = _select_sentences(
        sentences,
        {"must", "should", "required", "only", "cannot", "valid", "mandatory"},
        fallback_pool,
    )
    edges = _select_sentences(
        sentences,
        {"error", "empty", "blank", "missing", "invalid", "duplicate", "failure"},
        fallback_pool,
    )

    # Format the extracted fragments into a markdown-style report.
    sections = [
        "Features:",
        *[f"- {item}" for item in features],
        "",
        "User Flows:",
        *[f"- {item}" for item in flows],
        "",
        "Business Rules:",
        *[f"- {item}" for item in rules],
        "",
        "Edge Conditions:",
        *[f"- {item}" for item in edges],
    ]
    return "\n".join(sections)


def _generic_steps(subject: str, valid: bool = True) -> list[str]:
    """Provides a list of common QA test steps based on the inferred subject."""
    if "login" in subject:
        if valid:
            return [
                "Open the login page.",
                "Enter a valid username or email.",
                "Enter a valid password.",
                "Click the login button.",
            ]
        return [
            "Open the login page.",
            "Enter invalid or incomplete login credentials.",
            "Click the login button.",
        ]

    # Standard steps for any other type of feature.
    if valid:
        return [
            "Open the relevant page or workflow.",
            "Enter valid data in all required fields.",
            "Submit the request.",
        ]

    return [
        "Open the relevant page or workflow.",
        "Enter invalid, incomplete, or boundary-value data.",
        "Submit the request.",
    ]


def _format_case(
    case_id: str,
    title: str,
    steps: list[str],
    expected: str,
    test_type: str = "",
    coverage_dimensions: list[str] | None = None,
) -> str:
    """Helper to format a single test case block in standard format."""
    lines = [
        f"Test Case ID: {case_id}",
        f"Title: {title}",
        "Steps:",
        *[f"{index}. {step}" for index, step in enumerate(steps, start=1)],
        f"Expected Result: {expected}",
    ]
    if test_type:
        lines.append(f"Test Type: {test_type}")
    if coverage_dimensions:
        lines.append(f"Coverage Dimensions: {', '.join(coverage_dimensions)}")
    return "\n".join(lines)


def build_test_cases(analysis: str) -> str:
    """
    Acts as a 'Mock AI Test Generator'.
    It takes the local analysis and generates a broader set of QA test cases
    customized with the inferred subject of the PRD.
    """
    subject = infer_subject(analysis)

    cases = [
        _format_case(
            "TC-001",
            f"Verify successful {subject}",
            _generic_steps(subject, valid=True),
            f"The system completes the {subject} flow successfully and shows a success state.",
            "Happy Path",
            ["Happy Path"],
        ),
        _format_case(
            "TC-002",
            f"Verify required field validation for {subject}",
            ["Open the page.", "Leave mandatory fields blank.", "Submit the request."],
            "Mandatory fields are highlighted and clear validation messages are shown.",
            "Negative Scenarios",
            ["Negative Scenarios", "UI/UX Validation"],
        ),
        _format_case(
            "TC-003",
            f"Verify invalid data handling for {subject}",
            _generic_steps(subject, valid=False),
            "The request is rejected and the user sees a helpful validation error.",
            "Negative Scenarios",
            ["Negative Scenarios"],
        ),
        _format_case(
            "TC-004",
            f"Verify business rule enforcement for {subject}",
            ["Open the app.", "Enter data violating a business rule.", "Submit."],
            "The system blocks the action and explains which rule was violated.",
            "Negative Scenarios",
            ["Negative Scenarios", "State-Based Testing"],
        ),
        _format_case(
            "TC-005",
            f"Verify boundary and edge conditions for {subject}",
            ["Open the UI.", "Use extreme boundary-value input data.", "Submit."],
            "The system handles edge-case input safely without incorrect behavior.",
            "Boundary & Edge Cases",
            ["Boundary & Edge Cases"],
        ),
        _format_case(
            "TC-006",
            f"Verify error messaging and recovery for {subject}",
            ["Trigger failure.", "Observe the message.", "Retry with correct data."],
            "The user sees a clear error message and can recover successfully.",
            "Integration & Dependency Failures",
            ["Negative Scenarios", "Integration & Dependency Failures"],
        ),
        _format_case(
            "TC-007",
            f"Verify API success scenario for {subject}",
            ["Send valid API request.", "Inspect status and body.", "Verify DB."],
            "The API returns the expected success status, schema, and business data.",
            "Integration & Dependency Failures",
            ["Happy Path", "Integration & Dependency Failures", "Data Persistence & Consistency"],
        ),
        _format_case(
            "TC-008",
            f"Verify API validation failure for {subject}",
            ["Send API request with missing fields.", "Inspect status."],
            "The API returns a validation error and rejects the request safely.",
            "Negative Scenarios",
            ["Negative Scenarios", "Integration & Dependency Failures"],
        ),
        _format_case(
            "TC-009",
            f"Verify UI rendering and usability for {subject}",
            ["Open browser.", "Check labels and buttons.", "Check mobile view."],
            "The UI is usable, consistent, and responsive across viewports.",
            "UI/UX Validation",
            ["UI/UX Validation"],
        ),
        _format_case(
            "TC-010",
            f"Verify access control restrictions for {subject}",
            ["Open the restricted feature without meeting required permissions or state.", "Attempt the action."],
            "The system blocks access and explains why the action is not currently allowed.",
            "State-Based Testing",
            ["Negative Scenarios", "State-Based Testing", "UI/UX Validation"],
        ),
        _format_case(
            "TC-011",
            f"Verify persisted data remains consistent after refresh for {subject}",
            ["Complete the workflow successfully.", "Refresh or re-open the application.", "Review the saved data."],
            "The saved state remains intact and matches what was previously submitted.",
            "Data Persistence & Consistency",
            ["Data Persistence & Consistency"],
        ),
        _format_case(
            "TC-012",
            f"Verify invalid state transition is blocked for {subject}",
            ["Move the workflow into a completed or locked state.", "Attempt an action that should no longer be allowed."],
            "The invalid transition is blocked and an appropriate message is shown.",
            "State-Based Testing",
            ["State-Based Testing", "Negative Scenarios"],
        ),
        _format_case(
            "TC-013",
            f"Verify repeated actions do not duplicate {subject}",
            ["Open the workflow.", "Trigger the same action rapidly multiple times."],
            "The system processes the action safely without duplicate side effects.",
            "User Behavior Scenarios",
            ["User Behavior Scenarios", "Integration & Dependency Failures"],
        ),
        _format_case(
            "TC-014",
            f"Verify interrupted flow recovery for {subject}",
            ["Start the workflow.", "Refresh the page or navigate away mid-process.", "Return to the workflow."],
            "The user can recover predictably without corrupting state or losing valid data.",
            "User Behavior Scenarios",
            ["User Behavior Scenarios", "Data Persistence & Consistency"],
        ),
        _format_case(
            "TC-015",
            f"Verify dependency timeout handling for {subject}",
            ["Trigger the workflow while a downstream dependency is slow or unavailable."],
            "The system fails gracefully, preserves integrity, and guides the user to retry safely.",
            "Integration & Dependency Failures",
            ["Integration & Dependency Failures", "Negative Scenarios"],
        ),
        _format_case(
            "TC-016",
            f"Verify baseline performance for {subject}",
            ["Execute the workflow under normal load.", "Measure the response time."],
            "The workflow completes within the expected baseline performance threshold.",
            "Non-Functional Testing",
            ["Non-Functional Testing"],
        ),
        _format_case(
            "TC-017",
            f"Verify unauthorized or unsafe input is rejected for {subject}",
            ["Submit malformed, suspicious, or unauthorized input through the workflow."],
            "The system rejects the request safely without exposing sensitive details.",
            "Non-Functional Testing",
            ["Non-Functional Testing", "Negative Scenarios"],
        ),
        _format_case(
            "TC-018",
            f"Verify accessibility and cross-browser compatibility for {subject}",
            ["Open the workflow in multiple supported browsers and with keyboard-only navigation."],
            "The workflow remains usable, accessible, and visually correct across supported environments.",
            "UI/UX Validation",
            ["UI/UX Validation", "Non-Functional Testing"],
        ),
        _format_case(
            "TC-019",
            f"Verify database record integrity for {subject}",
            ["Complete the workflow.", "Inspect the persisted backend record or query result."],
            "The backend record accurately reflects the final workflow state and business data.",
            "Data Persistence & Consistency",
            ["Data Persistence & Consistency", "Integration & Dependency Failures"],
        ),
        _format_case(
            "TC-020",
            f"Verify regression safety for existing {subject} behavior",
            ["Complete a historically stable workflow related to the feature.", "Compare the behavior with the expected baseline."],
            "Existing behavior continues to work without regression after the new change.",
            "Regression Coverage",
            ["Regression Coverage"],
        ),
        _format_case(
            "TC-021",
            f"Verify true smoke health for {subject}",
            ["Open the application or primary system entry point.", "Check the minimal health or readiness signal.", "Confirm the primary workflow can start."],
            "The system proves minimal operational readiness and the primary workflow is reachable.",
            "Happy Path",
            ["Happy Path"],
        ),
        _format_case(
            "TC-022",
            f"Verify basic navigation smoke path for {subject}",
            ["Open the main landing or dashboard page.", "Navigate to the first page needed to start the workflow.", "Confirm controls and routing work."],
            "Basic navigation succeeds and the user can reach the core workflow entry path without blockers.",
            "UI/UX Validation",
            ["Happy Path", "UI/UX Validation"],
        ),
        _format_case(
            "TC-023",
            f"Verify core API health and reachability for {subject}",
            ["Call the health or readiness endpoint supporting the workflow.", "Inspect status and body.", "Confirm the service is reachable."],
            "The backend exposes a healthy ready state and the primary service path is reachable.",
            "Integration & Dependency Failures",
            ["Happy Path", "Integration & Dependency Failures"],
        ),
        _format_case(
            "TC-024",
            f"Verify API authentication and authorization rejection for {subject}",
            ["Send a protected API request without valid auth or with insufficient privileges.", "Inspect the response and resulting state."],
            "The API blocks unauthorized access with the correct auth failure behavior and no state change is persisted.",
            "Negative Scenarios",
            ["Negative Scenarios", "Integration & Dependency Failures", "Non-Functional Testing"],
        ),
        _format_case(
            "TC-025",
            f"Verify database default values and schema correctness for {subject}",
            ["Create the entity with only minimum required data.", "Inspect the stored record or backend representation."],
            "Default values, schema constraints, and stored field types are correct at the database layer.",
            "Data Persistence & Consistency",
            ["Data Persistence & Consistency"],
        ),
        _format_case(
            "TC-026",
            f"Verify backend access-control enforcement for {subject} data",
            ["Attempt to read or mutate restricted backend data with insufficient permission.", "Inspect the backend response and stored records."],
            "Unauthorized data access is blocked at the backend and no improper record mutation occurs.",
            "Data Persistence & Consistency",
            ["Data Persistence & Consistency", "Negative Scenarios", "Non-Functional Testing"],
        ),
        _format_case(
            "TC-027",
            f"Verify system-wide load and scalability for {subject}",
            ["Run the primary entry flow and common workflows under increasing concurrent user load.", "Monitor latency, errors, and throughput."],
            "The platform remains stable under load and scales without unacceptable degradation across core workflows.",
            "Non-Functional Testing",
            ["Non-Functional Testing"],
        ),
        _format_case(
            "TC-028",
            f"Verify dashboard or core-surface SLA for {subject}",
            ["Load the primary dashboard, landing page, or core screen used by the workflow.", "Measure response time against the target SLA."],
            "The core user-facing surface meets the expected response-time SLA under normal operating conditions.",
            "Non-Functional Testing",
            ["Non-Functional Testing", "UI/UX Validation"],
        ),
        _format_case(
            "TC-029",
            f"Verify endurance and compatibility for {subject}",
            ["Sustain realistic platform activity for a long-running session.", "Repeat the workflow across supported browsers or devices."],
            "The platform stays stable over long sessions and remains compatible across supported environments.",
            "Non-Functional Testing",
            ["Non-Functional Testing", "UI/UX Validation"],
        ),
        _format_case(
            "TC-030",
            f"Verify data leakage and auth-bypass resistance for {subject}",
            ["Attempt unauthorized data access, token tampering, or role escalation around the workflow.", "Inspect the returned data and resulting system state."],
            "Sensitive data is not exposed, auth bypass is blocked, and role boundaries remain enforced.",
            "Non-Functional Testing",
            ["Non-Functional Testing", "Negative Scenarios"],
        ),
        _format_case(
            "TC-031",
            f"Verify rare edge case recovery for {subject}",
            ["Trigger an unusual but plausible sequence of inputs, timing, or state conditions around the workflow.", "Observe the resulting behavior and recovery path."],
            "The system handles the rare edge case safely without corruption, silent failure, or inconsistent user state.",
            "Edge Case",
        ),
    ]

    return "\n\n".join(cases)
