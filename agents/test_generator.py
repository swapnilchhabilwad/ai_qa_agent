from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from langchain_community.callbacks import get_openai_callback

from utils.config import (
    get_azure_api_version,
    get_azure_endpoint,
    get_chat_model_name,
    get_required_env,
    has_valid_openai_key,
)
from utils.local_fallback import build_test_cases, infer_subject
from utils.test_suite import (
    CATEGORY_KEYWORDS,
    COVERAGE_CATEGORIES,
    CoverageCategoryReport,
    CoverageReport,
    TestCase,
    TestSuiteResult,
    assign_test_case_ids,
    calculate_coverage_score,
    clean_list_item,
    deduplicate_test_cases,
    infer_primary_dimension,
    keyword_matches_text,
    normalize_dimension_name,
    normalize_text,
    parse_test_cases,
    render_final_report,
)
from utils.usage_tracker import usage_tracker

LLM_RUNTIME_UNAVAILABLE = False

OUTPUT_TEST_CATEGORIES = [
    "Foundational Smoke and Sanity",
    "End-to-End Business Workflows",
    "Core Functional Requirements",
    "Detailed Functional Rules and Constraints",
    "Non-Functional Expectations",
    "Front-end UI and UX",
    "Front-end Responsive and Cross-Browser",
    "Back-end Database Persistence and Integrity",
    "Back-end Schema and Audit Validation",
    "API Contract and Schema Validation",
    "API CRUD and State Change Correctness",
    "Positive Happy Path Scenarios",
    "Negative Error Handling and Validation Scenarios",
    "State Transitions and Lifecycle flows",
    "User Behavior",
    "Concurrency and Race Conditions",
    "Interruption and Recovery scenarios",
    "Boundary and Edge Cases",
    "System Performance and Latency SLA",
    "Security, Auth, and Role-Based Access Control",
]

CATEGORY_SLICE_GUIDANCE = {
    "Foundational Smoke and Sanity": (
        "Focus on true smoke checks, minimal working system validation, core health checks, "
        "and basic reachability of the primary entry points."
    ),
    "End-to-End Business Workflows": (
        "Focus on the most important complete business journeys and end-to-end flows across the system."
    ),
    "Core Functional Requirements": (
        "Focus on core feature behavior and primary functional expectations stated in the requirements."
    ),
    "Detailed Functional Rules and Constraints": (
        "Focus on specific business rules, constraints, entitlement logic, and role-aware functional correctness."
    ),
    "Non-Functional Expectations": (
        "Focus on feature-level non-functional expectations like reliability, availability, and basic error resilience."
    ),
    "Front-end UI and UX": (
        "Focus on UI/UX behavior, labels, buttons, copy accuracy, layout consistency, and usability on primary screens."
    ),
    "Front-end Responsive and Cross-Browser": (
        "Focus on mobile view, tablet view, responsiveness, and behavior across different supported browsers."
    ),
    "Back-end Database Persistence and Integrity": (
        "Focus on data persistence, stored values, and data integrity after system actions."
    ),
    "Back-end Schema and Audit Validation": (
        "Focus on schema correctness, default values, audit trails, and backend record structures."
    ),
    "API Contract and Schema Validation": (
        "Focus on API contract adherence, request/response schema validation, and status code correctness."
    ),
    "API CRUD and State Change Correctness": (
        "Focus on API-driven CRUD operations, state changes, and business logic enforcement at the service layer."
    ),
    "Positive Happy Path Scenarios": (
        "Focus on clean successful flows where everything works as intended with valid inputs."
    ),
    "Negative Error Handling and Validation Scenarios": (
        "Focus on invalid input handling, rejected actions, and clear error feedback for the user."
    ),
    "State Transitions and Lifecycle flows": (
        "Focus on lifecycle/state transitions and transitions between valid and invalid system states."
    ),
    "User Behavior": (
        "Focus on user-driven patterns like page refresh, back navigation, and multi-tab interactions."
    ),
    "Concurrency and Race Conditions": (
        "Focus on repeated actions, concurrent requests, and user-driven race conditions."
    ),
    "Interruption and Recovery scenarios": (
        "Focus on system interruptions, timeouts, and recovery paths for the user."
    ),
    "Boundary and Edge Cases": (
        "Focus on min/max thresholds, empty states, nulls, and unusual but realistic edge conditions."
    ),
    "System Performance and Latency SLA": (
        "Focus on platform load, response-time SLA, latency, and scalability across core workflows."
    ),
    "Security, Auth, and Role-Based Access Control": (
        "Focus on authentication, authorization bypass attempts, data leakage prevention, and role-based access validation."
    ),
}

OUTPUT_CATEGORY_BY_DIMENSION = {
    "Happy Path": "Positive Happy Path Scenarios",
    "Negative Scenarios": "Negative Error Handling and Validation Scenarios",
    "Boundary & Edge Cases": "Boundary and Edge Cases",
    "State-Based Testing": "State Transitions and Lifecycle flows",
    "User Behavior Scenarios": "User Behavior",
    "Integration & Dependency Failures": "Interruption and Recovery scenarios",
    "UI/UX Validation": "Front-end UI and UX",
    "Non-Functional Testing": "Non-Functional Expectations",
    "Data Persistence & Consistency": "Back-end Database Persistence and Integrity",
    "Regression Coverage": "Core Functional Requirements",
}

SUPPLEMENTAL_QUALITY_GATES = [
    {
        "name": "True smoke health and minimal working system coverage",
        "owner": "Happy Path",
        "output_categories": ["Foundational Smoke and Sanity"],
        "keywords": ("smoke", "sanity", "health", "minimal working system", "availability", "heartbeat"),
        "min_cases": 1,
        "missing_scenario": (
            "Add true smoke validation for minimal working system behavior or a basic system health check."
        ),
        "untested_risk": (
            "Critical availability issues may survive into deeper testing because the suite lacks a foundational smoke gate."
        ),
    },
    {
        "name": "Smoke entry and basic navigation coverage",
        "owner": "Happy Path",
        "output_categories": ["Foundational Smoke and Sanity"],
        "keywords": (
            "login",
            "signin",
            "sign in",
            "authentication",
            "entry flow",
            "landing page",
            "home page",
            "basic navigation",
            "dashboard",
            "navigate",
        ),
        "min_cases": 2,
        "missing_scenario": (
            "Add foundational smoke cases for primary entry or authentication flow and basic navigation across core surfaces."
        ),
        "untested_risk": (
            "The suite may miss broken primary access paths because smoke coverage does not validate entry and navigation clearly."
        ),
    },
    {
        "name": "Core API health and reachability coverage",
        "owner": "Integration & Dependency Failures",
        "output_categories": [
            "Foundational Smoke and Sanity",
            "API Contract and Schema Validation",
        ],
        "keywords": ("api health", "service health", "health endpoint", "status endpoint", "ping", "reachability"),
        "min_cases": 1,
        "missing_scenario": (
            "Add core API or service health coverage so smoke validation includes basic backend reachability."
        ),
        "untested_risk": (
            "Backend availability regressions may go unnoticed because the suite does not validate core API health."
        ),
    },
    {
        "name": "API success, contract, and CRUD correctness coverage",
        "owner": "Integration & Dependency Failures",
        "output_categories": ["API CRUD and State Change Correctness"],
        "keywords": (
            "api",
            "endpoint",
            "contract",
            "schema",
            "status code",
            "201",
            "200",
            "crud",
            "create",
            "update",
            "delete",
            "idempotent",
        ),
        "min_cases": 3,
        "missing_scenario": (
            "Add API correctness coverage for success responses, contract/schema checks, and core CRUD or state-change behavior."
        ),
        "untested_risk": (
            "Service behavior may be incorrect even if UI flows pass because the suite lacks direct API contract coverage."
        ),
    },
    {
        "name": "API validation, auth, and authorization failure coverage",
        "owner": "Negative Scenarios",
        "output_categories": ["API Contract and Schema Validation"],
        "keywords": (
            "api",
            "endpoint",
            "400",
            "401",
            "403",
            "validation error",
            "missing field",
            "unauthorized",
            "forbidden",
            "auth token",
        ),
        "min_cases": 2,
        "missing_scenario": (
            "Add negative API coverage for validation failures, authentication failures, and authorization rejection."
        ),
        "untested_risk": (
            "Invalid or unauthorized API calls may behave incorrectly because failure-path API coverage is too shallow."
        ),
    },
    {
        "name": "DB-level correctness and persistence confirmation coverage",
        "owner": "Data Persistence & Consistency",
        "output_categories": ["Back-end Database Persistence and Integrity"],
        "keywords": (
            "database",
            "db",
            "schema",
            "stored",
            "persisted",
            "audit",
            "default value",
            "backend record",
            "query result",
        ),
        "min_cases": 3,
        "missing_scenario": (
            "Add DB-level validation for persisted values, schema or default-value correctness, and backend record accuracy."
        ),
        "untested_risk": (
            "System correctness may be unverified because backend state and schema behavior are not tested directly."
        ),
    },
    {
        "name": "Backend access-control enforcement coverage",
        "owner": "Data Persistence & Consistency",
        "output_categories": ["Back-end Database Persistence and Integrity"],
        "keywords": (
            "access control",
            "role-based",
            "permission",
            "entitlement",
            "restricted record",
            "unauthorized data access",
            "db-level",
        ),
        "min_cases": 1,
        "missing_scenario": (
            "Add backend or DB-layer access-control validation so restricted data cannot be read or mutated incorrectly."
        ),
        "untested_risk": (
            "Data protection gaps may remain hidden because access-control enforcement is not validated below the UI layer."
        ),
    },
    {
        "name": "System security coverage for data leakage and unauthorized access",
        "owner": "Non-Functional Testing",
        "output_categories": ["Security, Auth, and Role-Based Access Control"],
        "keywords": (
            "security",
            "data leakage",
            "unauthorized data access",
            "sensitive data",
            "exposure",
            "leak",
        ),
        "min_cases": 2,
        "missing_scenario": (
            "Add true security testing for data leakage prevention and unauthorized data access attempts."
        ),
        "untested_risk": (
            "Sensitive information could be exposed because the suite lacks explicit data leakage and unauthorized-access coverage."
        ),
    },
    {
        "name": "System security coverage for auth bypass and role-based access",
        "owner": "Non-Functional Testing",
        "output_categories": ["Security, Auth, and Role-Based Access Control"],
        "keywords": (
            "auth bypass",
            "authorization bypass",
            "role-based",
            "rbac",
            "privilege escalation",
            "token tamper",
            "session hijack",
        ),
        "min_cases": 2,
        "missing_scenario": (
            "Add true security testing for authentication bypass, authorization bypass, and role-based access enforcement."
        ),
        "untested_risk": (
            "Privilege-escalation or bypass issues may be missed because the suite does not explicitly test auth and role boundaries."
        ),
    },
    {
        "name": "System-wide load, SLA, and scalability coverage",
        "owner": "Non-Functional Testing",
        "output_categories": ["System Performance and Latency SLA"],
        "keywords": (
            "load",
            "stress",
            "sla",
            "response time",
            "latency",
            "throughput",
            "dashboard load",
            "registration under load",
            "scalability",
            "concurrent users",
        ),
        "min_cases": 3,
        "missing_scenario": (
            "Add full-system performance coverage for load, response-time SLA, and scalability on primary platform workflows."
        ),
        "untested_risk": (
            "Performance issues may remain hidden because the suite is not validating load, latency, and scalability across the system."
        ),
    },
    {
        "name": "System-wide endurance and compatibility coverage",
        "owner": "Non-Functional Testing",
        "output_categories": ["Non-Functional Expectations"],
        "keywords": (
            "endurance",
            "soak",
            "long-running",
            "8 hour",
            "compatibility",
            "cross-browser",
            "browser compatibility",
            "device compatibility",
        ),
        "min_cases": 2,
        "missing_scenario": (
            "Add long-running endurance and cross-environment compatibility coverage for the full platform."
        ),
        "untested_risk": (
            "Long-session stability and compatibility regressions may be missed because endurance and compatibility coverage are shallow."
        ),
    },
]

OUTPUT_CATEGORY_ORDER = {
    category: index for index, category in enumerate(OUTPUT_TEST_CATEGORIES)
}


def _llm_enabled() -> bool:
    return has_valid_openai_key() and not LLM_RUNTIME_UNAVAILABLE


def _invoke_llm(prompt: str, task_label: str, temperature: float = 0) -> str:
    global LLM_RUNTIME_UNAVAILABLE

    if not _llm_enabled():
        return ""

    try:
        with get_openai_callback() as cb:
            # Note: _get_llm is lru_cached, but it will cache separately for each temperature.
            response = _get_llm(temperature).invoke(prompt).content
            usage_tracker.record_usage(task_label, cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)
            return response
    except Exception as exc:
        LLM_RUNTIME_UNAVAILABLE = True
        print(f"{task_label} failed ({type(exc).__name__}).")
        return ""


def _extract_json_payload(response: str):
    cleaned = response.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    candidates: list[str] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _split_to_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_split_to_list(item))
        return [item for item in items if item]

    text = str(value).strip()
    if not text:
        return []

    raw_parts = re.split(r"\n+|;|(?=\d+[.)]\s)", text)
    cleaned_parts = [clean_list_item(part) for part in raw_parts if clean_list_item(part)]
    return cleaned_parts if cleaned_parts else [text]


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value.strip())
    return deduped


def _category_focus(category: str) -> str:
    return CATEGORY_SLICE_GUIDANCE.get(
        category,
        "Cover the requested slice comprehensively while keeping category taxonomy intact.",
    )


def _preferred_output_category_for_dimension(dimension: str) -> str:
    return OUTPUT_CATEGORY_BY_DIMENSION.get(dimension, "Functional and Non-functional")


def _determine_output_category(test_case: TestCase, fallback_category: str) -> str:
    if test_case.category in OUTPUT_TEST_CATEGORIES:
        return test_case.category

    if fallback_category in OUTPUT_TEST_CATEGORIES and test_case.category in ("", "General"):
        fallback_allowed = fallback_category
    else:
        fallback_allowed = ""

    haystack = normalize_text(
        " ".join(
            [
                fallback_category,
                test_case.category,
                test_case.title,
                test_case.test_type,
                " ".join(test_case.coverage_dimensions),
                " ".join(test_case.steps[:4]),
                test_case.expected_result,
            ]
        )
    )

    specialized_checks = [
        ("Edge case", ("edge case", "rare condition", "corner case")),
        (
            "API contract, CRUD, integration, and service behavior",
            ("api", "endpoint", "contract", "crud", "status code", "service behavior", "service health"),
        ),
        (
            "Back-end database, persistence, schema, and audit validation",
            ("database", "schema", "audit", "persistence", "db", "stored", "backend record"),
        ),
        (
            "System-wide performance, security, reliability, scalability, endurance, and compatibility",
            (
                "performance",
                "security",
                "reliability",
                "scalability",
                "endurance",
                "compatibility",
                "load",
                "stress",
                "sla",
                "latency",
                "data leakage",
                "auth bypass",
                "unauthorized data access",
            ),
        ),
    ]

    for category, keywords in specialized_checks:
        if any(keyword_matches_text(haystack, keyword) for keyword in keywords):
            return category

    preferred_dimension_categories = [
        "Front-end UI, UX",
        "Back-end database, persistence, schema, and audit validation",
        "State transitions, user behavior, concurrency, and interruption scenarios",
        "Integration, billing, payment, third-party dependency, and failure-recovery scenarios",
        "Boundary, edge, empty-state, and data-combination",
        "Edge case",
        "Positive, negative, and business-rule validation",
        "Foundational smoke, sanity, and end-to-end business workflow",
        "Functional and Non-functional",
    ]

    dimension_matches: list[str] = []
    for dimension in test_case.coverage_dimensions:
        normalized_dimension = normalize_dimension_name(dimension)
        preferred = OUTPUT_CATEGORY_BY_DIMENSION.get(normalized_dimension)
        if preferred and preferred not in dimension_matches:
            dimension_matches.append(preferred)

    for category in preferred_dimension_categories:
        if category in dimension_matches:
            return category

    checks = [
        (
            "Front-end UI, UX",
            ("ui", "ux", "accessibility", "browser", "navigation", "screen", "layout", "visibility"),
        ),
        (
            "State transitions, user behavior, concurrency, and interruption scenarios",
            (
                "state",
                "transition",
                "concurrency",
                "interrupt",
                "double click",
                "refresh",
                "back navigation",
                "multi-tab",
                "user behavior",
            ),
        ),
        (
            "Integration, billing, payment, third-party dependency, and failure-recovery scenarios",
            ("integration", "billing", "payment", "third-party", "dependency", "retry", "rollback", "timeout"),
        ),
        (
            "Boundary, edge, empty-state, and data-combination",
            ("boundary", "empty-state", "empty state", "data-combination", "combination", "minimum", "maximum"),
        ),
        ("Edge case", ("edge case", "rare condition", "corner case")),
        (
            "Positive, negative, and business-rule validation",
            ("positive", "negative", "business rule", "validation", "rejected", "invalid"),
        ),
        (
            "Foundational smoke, sanity, and end-to-end business workflow",
            ("smoke", "sanity", "end-to-end", "health check", "minimal working system", "basic navigation"),
        ),
        (
            "Functional and Non-functional",
            ("functional", "non-functional", "access control", "eligibility", "entitlement", "role-based", "plan lifecycle"),
        ),
    ]

    for category, keywords in checks:
        if any(keyword_matches_text(haystack, keyword) for keyword in keywords):
            return category

    if fallback_allowed:
        return fallback_allowed

    return "Functional and Non-functional"


def _case_haystack(test_case: TestCase) -> str:
    return normalize_text(
        " ".join(
            [
                test_case.category,
                test_case.title,
                test_case.test_type,
                " ".join(test_case.coverage_dimensions),
                " ".join(test_case.preconditions),
                " ".join(test_case.steps),
                test_case.expected_result,
            ]
        )
    )


def _case_matches_quality_gate(test_case: TestCase, gate: dict) -> bool:
    if gate.get("output_categories") and test_case.category not in gate["output_categories"]:
        return False

    haystack = _case_haystack(test_case)
    return any(keyword_matches_text(haystack, keyword) for keyword in gate.get("keywords", ()))


def _append_reason(reason: str, extra: str) -> str:
    cleaned_reason = reason.strip()
    cleaned_extra = extra.strip()
    if not cleaned_extra or cleaned_extra in cleaned_reason:
        return cleaned_reason
    if not cleaned_reason:
        return cleaned_extra
    return f"{cleaned_reason} {cleaned_extra}"


def _apply_supplemental_quality_gates(
    report: CoverageReport,
    test_cases: list[TestCase],
) -> CoverageReport:
    report_by_category = {item.category: item for item in report.categories}

    for gate in SUPPLEMENTAL_QUALITY_GATES:
        owner = gate["owner"]
        report_item = report_by_category.get(owner)
        if not report_item:
            continue

        matched_cases = [
            test_case
            for test_case in test_cases
            if _case_matches_quality_gate(test_case, gate)
        ]
        if len(matched_cases) >= gate["min_cases"]:
            continue

        detail = (
            f"{gate['name']} is missing."
            if not matched_cases
            else f"{gate['name']} is too shallow with only {len(matched_cases)} matching test case(s)."
        )
        report_item.status = (
            report_item.status
            if STATUS_RANK[report_item.status] < STATUS_RANK["partial"]
            else "partial"
        )
        report_item.reason = _append_reason(report_item.reason, detail)
        report_item.evidence_ids = _dedupe_strings(
            report_item.evidence_ids + [test_case.case_id for test_case in matched_cases if test_case.case_id]
        )
        report_item.missing_scenarios = _dedupe_strings(
            report_item.missing_scenarios + [gate["missing_scenario"]]
        )
        report_item.untested_risks = _dedupe_strings(
            report_item.untested_risks + [gate["untested_risk"]]
        )
        report.missing_scenarios = _dedupe_strings(
            report.missing_scenarios + [gate["missing_scenario"]]
        )
        report.untested_risks = _dedupe_strings(
            report.untested_risks + [gate["untested_risk"]]
        )
        report.weak_coverage_areas = _dedupe_strings(
            report.weak_coverage_areas + [f"{owner}: {detail}"]
        )

    return report


def _sort_test_cases_for_output(test_cases: list[TestCase]) -> list[TestCase]:
    return sorted(
        test_cases,
        key=lambda test_case: (
            OUTPUT_CATEGORY_ORDER.get(test_case.category, len(OUTPUT_CATEGORY_ORDER)),
            test_case.source_pass,
            test_case.case_id or "",
            normalize_text(test_case.title),
        ),
    )


def _infer_dimensions_from_case(test_case: TestCase) -> list[str]:
    haystack = normalize_text(
        " ".join(
            [
                test_case.category,
                test_case.title,
                test_case.test_type,
                " ".join(test_case.steps),
                test_case.expected_result,
            ]
        )
    )

    matched = [
        category
        for category in COVERAGE_CATEGORIES
        if normalize_text(category) in haystack
        or any(keyword_matches_text(haystack, keyword) for keyword in CATEGORY_KEYWORDS.get(category, ()))
    ]

    if not matched:
        matched = [infer_primary_dimension(test_case)]

    return _dedupe_strings(matched)


def _decorate_test_cases(test_cases: list[TestCase], default_category: str, source_pass: str) -> list[TestCase]:
    for test_case in test_cases:
        test_case.category = _determine_output_category(test_case, default_category)
        test_case.source_pass = source_pass
        test_case.coverage_dimensions = _dedupe_strings(
            [normalize_dimension_name(value) for value in test_case.coverage_dimensions if value]
        )
        if not test_case.coverage_dimensions:
            test_case.coverage_dimensions = _infer_dimensions_from_case(test_case)
        if not test_case.test_type:
            test_case.test_type = infer_primary_dimension(test_case)

    return deduplicate_test_cases(test_cases)


def _coerce_test_case(raw_case, default_category: str, source_pass: str) -> TestCase | None:
    if not isinstance(raw_case, dict):
        return None

    title = str(raw_case.get("title") or raw_case.get("name") or "").strip()
    expected_result = str(
        raw_case.get("expected_result")
        or raw_case.get("expected")
        or raw_case.get("expected_outcome")
        or ""
    ).strip()
    steps = _split_to_list(raw_case.get("steps"))

    if not title or not steps or not expected_result:
        return None

    return TestCase(
        case_id=str(raw_case.get("test_case_id") or raw_case.get("id") or "").strip(),
        title=title,
        preconditions=_split_to_list(raw_case.get("preconditions") or raw_case.get("prerequisites")),
        steps=steps,
        expected_result=expected_result,
        test_type=str(raw_case.get("test_type") or raw_case.get("type") or "").strip(),
        category=str(raw_case.get("category") or default_category).strip() or default_category,
        source_pass=str(raw_case.get("generation_source") or source_pass).strip() or source_pass,
        coverage_dimensions=_split_to_list(
            raw_case.get("coverage_dimensions") or raw_case.get("coverage_categories")
        ),
    )


def _parse_generation_response(response: str, default_category: str, source_pass: str) -> list[TestCase]:
    payload = _extract_json_payload(response)
    parsed_cases: list[TestCase] = []

    if isinstance(payload, dict):
        raw_cases = payload.get("test_cases") or payload.get("cases") or []
        parsed_cases = [
            test_case
            for raw_case in raw_cases
            if (test_case := _coerce_test_case(raw_case, default_category, source_pass))
        ]
    elif isinstance(payload, list):
        parsed_cases = [
            test_case
            for raw_case in payload
            if (test_case := _coerce_test_case(raw_case, default_category, source_pass))
        ]

    if parsed_cases:
        return _decorate_test_cases(parsed_cases, default_category, source_pass)

    markdown_cases = parse_test_cases(response)
    if markdown_cases:
        return _decorate_test_cases(markdown_cases, default_category, source_pass)

    open("debug_raw_llm_output.txt", "a", encoding="utf-8").write(f"\n\n--- RAW ---\n{response}\n")
    print(f"\n[DEBUG] LLM Response failed to parse! Saved to debug_raw_llm_output.txt")
    return []


def _case_catalog(test_cases: list[TestCase], include_steps: bool = False) -> str:
    if not test_cases:
        return "None"

    lines: list[str] = []
    for test_case in test_cases:
        dimensions = ", ".join(test_case.coverage_dimensions) or infer_primary_dimension(test_case)
        line = (
            f"- {test_case.case_id or 'TBD'} | {test_case.title} | "
            f"Type: {test_case.test_type or infer_primary_dimension(test_case)} | "
            f"Dimensions: {dimensions} | Expected: {test_case.expected_result}"
        )
        lines.append(line)
        if include_steps:
            lines.append(f"  Steps: {' -> '.join(test_case.steps[:5])}")
    return "\n".join(lines)


def _filter_new_cases(candidate_cases: list[TestCase], existing_cases: list[TestCase]) -> list[TestCase]:
    existing_fingerprints = {test_case.fingerprint() for test_case in existing_cases}
    new_cases: list[TestCase] = []

    for test_case in candidate_cases:
        fingerprint = test_case.fingerprint()
        if fingerprint in existing_fingerprints:
            continue
        existing_fingerprints.add(fingerprint)
        new_cases.append(test_case)

    return new_cases


def _build_generation_prompt(
    context: str,
    category: str,
    impact_analysis: str,
    existing_cases: list[TestCase],
) -> str:
    regression_note = (
        f"\nImpact Analysis Matrix:\n{impact_analysis}\n"
        if impact_analysis and "Regression" in category
        else ""
    )

    return f"""
    You are a Senior QA Engineer.

    Generate net-new test cases for the generation slice below. Think critically and avoid shallow overlap.
    You must produce BOTH:
    - V1 breadth: foundational smoke tests, basic happy paths, core CRUD/API checks, DB validations, UI sanity, access control, and broad system coverage.
    - V2 depth: edge cases, failure handling, retries, race conditions, interruption flows, partial failures, dependency issues, and realistic user behavior.

    Mandatory coverage labels you may use in coverage_dimensions:
    {", ".join(COVERAGE_CATEGORIES)}

    Allowed output categories. Use these exact labels only:
    {", ".join(OUTPUT_TEST_CATEGORIES)}

    Rules:
    - Return plain MARKDOWN only. Do not use JSON.
    - Use exactly this markdown format for each test case:

      ### Category: [Allowed Output Category]
      Test Case ID: [Generate unique ID, e.g. TC-001]
      Title: [Clear test scenario]
      Preconditions:
      - [Precondition 1]
      Steps:
      - [Step 1]
      - [Step 2]
      Expected Result: [Expected behavior]
      Test Type: [e.g., Performance, Functional]
      Coverage Dimensions: [Label1], [Label2]
    - Generate as many distinct test cases as needed to cover this slice comprehensively.
    - DO NOT limit in generating test cases in any way. I want as many test cases as possible.
    - If a certain category is able to generate more than 50, 70, or 100+ test cases, be it so. Generate them all.
    - There is NO maximum limit. Exhaust every possible permutations and scenarios.
    - Each case must validate a unique behavior or risk.
    - Do not duplicate or paraphrase the existing suite.
    - Use only the coverage labels listed above.
    - For regression slices, generate only targeted regression cases from the impact matrix.
    - Prefer completeness over brevity. Include common, uncommon, and high-risk scenarios where they are relevant.
    - Include nuanced scenarios such as timing windows, retries, interrupted flows, partial failures, validation combinations, permission/state interactions, data integrity checks, API contract checks, DB validation checks, and compatibility/security checks when applicable.
    - Make the smoke slice a true smoke suite: minimal working system checks, primary entry or authentication flow when relevant, basic navigation, and core API/service reachability.
    - Make the API slice a true correctness suite: success status codes, validation failures, auth/authz failures, contract/schema checks, CRUD or state-change behavior, and persistence confirmation.
    - Make the system-wide non-functional slice platform-wide rather than feature-only: load, response-time SLA, scalability, endurance, compatibility, data leakage, auth bypass, and role-based access validation.
    - Do not skip foundational coverage just because advanced scenarios exist.
    - If the slice is broad, keep generating until no substantial distinct scenario family remains for that slice.
    - When access control, feature-gating, eligibility, entitlement, or lifecycle-state logic appears, place those tests under "Functional and Non-functional" so the category remains project-agnostic.
    - Do not invent product rules, permissions, or entitlements that are not stated or strongly implied in the requirements.
    - If a requirement states a direction of access control or entitlement, preserve it exactly. Do not invert it.
    - Avoid requirement interpretation drift. Prefer "verify stated rule" over "invent likely rule".

    Generation Slice:
    {category}

    Slice Guidance:
    {_category_focus(category)}

    Existing Test Cases To Avoid Duplicating:
    {_case_catalog(existing_cases)}

    Requirement Analysis:
    {context}
    {regression_note}
    """


def _build_generation_expansion_prompt(
    context: str,
    category: str,
    impact_analysis: str,
    existing_cases: list[TestCase],
) -> str:
    regression_note = (
        f"\nImpact Analysis Matrix:\n{impact_analysis}\n"
        if impact_analysis and "Regression" in category
        else ""
    )

    return f"""
    You are a Senior QA Engineer doing a second sweep on the same test slice.

    Generate only overlooked, rare, or deeper follow-up test cases for this slice.
    This sweep must strengthen BOTH breadth and depth by filling any missing foundational checks and advanced scenarios that were not covered in the first sweep.

    Mandatory coverage labels you may use in coverage_dimensions:
    {", ".join(COVERAGE_CATEGORIES)}

    Allowed output categories. Use these exact labels only:
    {", ".join(OUTPUT_TEST_CATEGORIES)}

    Rules:
    - Return plain MARKDOWN only using the same format as before.
    - Every generated test case must begin with `### Category: [Allowed Output Category]`
    - Generate as many additional test cases as needed for meaningful expansion of this slice.
    - DO NOT limit in generating test cases in any way. I want as many test cases as possible.
    - If a certain category is able to generate more than 50, 70, or 100+ test cases, be it so. Generate them all.
    - There is NO maximum limit. Exhaust every possible permutations and scenarios.
    - Do not repeat already covered happy paths or standard validations.
    - Focus on hidden risks: timing windows, retries, interruptions, race conditions, partial failure states, boundary combinations, permission-state combinations, rollback behavior, recovery paths, core API/DB gaps, security gaps, compatibility gaps, and smoke-flow omissions.
    - If the slice still lacks true smoke, core API correctness, DB-level validation, or full-system performance/security coverage, prioritize those before adding rarer variants.
    - Every case must be materially different from the existing suite.
    - Continue until the remaining new ideas are minor variants rather than meaningfully new risks.
    - Do not invent product rules, roles, or permissions beyond the requirement text.

    Generation Slice:
    {category}

    Slice Guidance:
    {_category_focus(category)}

    Existing Test Cases Already Covered:
    {_case_catalog(existing_cases, include_steps=True)}

    Requirement Analysis:
    {context}
    {regression_note}
    """


def _build_gap_generation_prompt(
    context: str,
    impact_analysis: str,
    existing_cases: list[TestCase],
    coverage_report: CoverageReport,
    source_pass: str,
) -> str:
    gap_lines: list[str] = []
    for category_report in coverage_report.categories:
        if category_report.status == "covered":
            continue
        missing = "; ".join(category_report.missing_scenarios) or "No explicit scenario listed."
        risks = "; ".join(category_report.untested_risks) or "No explicit risk listed."
        gap_lines.append(
            f"- {category_report.category} | Status: {category_report.status} | "
            f"Missing: {missing} | Risks: {risks}"
        )

    weak_areas = "\n".join(f"- {item}" for item in coverage_report.weak_coverage_areas) or "- None"
    impact_block = f"\nImpact Analysis Matrix:\n{impact_analysis}\n" if impact_analysis else ""

    return f"""
    You are a Senior QA Reviewer performing PASS 2 self-evaluation gap filling.

    Your job is to generate ONLY the missing or weak test coverage. Do not regenerate the suite.

    Mandatory coverage labels you may use in coverage_dimensions:
    {", ".join(COVERAGE_CATEGORIES)}

    Allowed output categories. Use these exact labels only:
    {", ".join(OUTPUT_TEST_CATEGORIES)}

    Rules:
    - Return plain MARKDOWN only using the same format as before.
    - Every generated test case must begin with `### Category: [Allowed Output Category]`
    - Generate only net-new test cases for the gaps listed below.
    - DO NOT limit in generating test cases in any way. I want as many test cases as possible.
    - If a certain category is able to generate more than 50, 70, or 100+ test cases, be it so. Generate them all.
    - There is NO maximum limit. Exhaust every possible permutations and scenarios.
    - Do not duplicate or restate existing tests.
    - Generate as many focused test cases as needed to close the identified gaps.
    - If a category is already covered, do not generate more tests for it.
    - Be strict and targeted.
    - Prefer nuanced gaps, not superficial rewrites of existing tests.
    - If foundational breadth is missing, add foundational coverage first before layering on advanced variants.
    - If the gaps mention smoke, API correctness, DB-level validation, security, or system-wide performance, generate direct tests for those concerns rather than more workflow-only scenarios.
    - Do not invent access rules, role behavior, or entitlement directions beyond the requirement text.

    Current Pass Label:
    {source_pass}

    Gaps To Fill:
    {chr(10).join(gap_lines) if gap_lines else "- None"}

    Weak Coverage Areas:
    {weak_areas}

    Existing Test Suite:
    {_case_catalog(existing_cases, include_steps=True)}

    Requirement Analysis:
    {context}
    {impact_block}
    """


def _build_coverage_evaluation_prompt(
    context: str,
    impact_analysis: str,
    test_cases: list[TestCase],
) -> str:
    impact_block = f"\nImpact Analysis Matrix:\n{impact_analysis}\n" if impact_analysis else ""

    return f"""
    You are a Senior QA Architect reviewing generated test cases.

    Evaluate coverage STRICTLY across these categories only:
    {", ".join(COVERAGE_CATEGORIES)}

    Your primary goal is to identify ANY and ALL remaining gaps, no matter how small, to ensure the MOST exhaustive test suite possible. Do not stop at "sufficient" coverage; strive for absolute completeness. The system will continue to generate tests as long as gaps are identified.

    Rules:
    - Return JSON only.
    - Use the schema:
      {{
        "categories": [
          {{
            "category": "Happy Path",
            "status": "covered",
            "reason": "...",
            "evidence_ids": ["TC-001", "TC-002"],
            "missing_scenarios": ["..."],
            "untested_risks": ["..."]
          }}
        ]
      }}
    - For each category, determine if it is "covered", "partial", or "missing".
    - A category is "covered" ONLY if it is truly exhaustive, addressing all possible scenarios, including edge cases, negative paths, and nuanced interactions. If there's any room for more tests, it's "partial".
    - Provide specific reasons for "partial" or "missing" statuses, detailing what scenarios are lacking or what risks remain untested.
    - List concrete `evidence_ids` for test cases that contribute to the coverage of each category.
    - Identify `missing_scenarios` and `untested_risks` for "partial" or "missing" categories.
    - Be extremely critical. Assume the current test cases are a first pass, and your job is to find every possible weakness.
    - Do not declare "covered" unless you are absolutely certain no more meaningful test cases can be generated for that specific category.
    - If a category has a low number of test cases, it is likely "partial" or "missing" unless the domain is extremely simple.
    - Consider the user's explicit request for "as many test cases as possible" and "do not limit in generating test cases in any way." Reflect this in your evaluation.

    Requirement Analysis:
    {context}
    {impact_block}

    Existing Test Cases:
    {_case_catalog(test_cases, include_steps=True)}
    

    Review rules:
    - A category cannot be "covered" without explicit evidence_ids.
    - If evidence is weak or narrow, mark it "partial".
    - If the suite does not clearly exercise the category, mark it "missing".
    - Do not hallucinate coverage.
    - If no impact matrix is provided, treat Regression Coverage as applicable only if the suite explicitly targets prior functionality; otherwise explain that regression is not applicable.
    - While scoring the required categories, also inspect whether the suite includes foundational smoke coverage, core API/CRUD checks, DB validation coverage, access-control breadth, and system-wide non-functional checks such as security, accessibility, browser/device compatibility, scalability, and endurance.
    - Do not consider smoke sufficient if it only contains deep end-to-end journeys. True smoke requires minimal working system validation, basic navigation or entry checks, and basic service/API reachability.
    - Do not consider API coverage sufficient if it lacks explicit success-code checks, validation failures, auth/authz failures, contract/schema checks, or backend state/persistence confirmation.
    - Do not consider security coverage sufficient unless the suite explicitly tests unauthorized data access, auth bypass, and role-based access validation.
    - Do not consider system-wide performance sufficient if it only covers one feature path. Expect platform-wide load, response-time SLA, scalability, endurance, and compatibility checks where applicable.
    - If foundational breadth is absent, the related category should be partial or missing even if a few advanced scenarios exist.
    - Do not reward tests that rely on invented permissions, invented role behavior, or unsupported entitlement assumptions.

    Return JSON only in this exact shape:
    {{
      "categories": [
        {{
          "category": "Happy Path",
          "status": "covered",
          "reason": "...",
          "evidence_ids": ["HP-001"],
          "missing_scenarios": ["..."],
          "untested_risks": ["..."]
        }}
      ],
      "missing_scenarios": ["..."],
      "untested_risks": ["..."],
      "weak_coverage_areas": ["..."]
    }}

    Requirement Analysis:
    {context}
    {impact_block}

    Current Test Suite:
    {_case_catalog(test_cases, include_steps=True)}
    """


def _normalize_status(value: str) -> str:
    normalized = normalize_text(value)
    if "cover" in normalized and "partial" not in normalized:
        return "covered"
    if "partial" in normalized or "weak" in normalized:
        return "partial"
    return "missing"


def _category_missing_hint(category: str, subject: str) -> str:
    hints = {
        "Happy Path": f"End-to-end successful {subject} flow with the expected success confirmation.",
        "Negative Scenarios": f"Invalid or disallowed {subject} action is rejected with clear user feedback.",
        "Boundary & Edge Cases": f"{subject} behavior at min, max, empty, null, and threshold values.",
        "State-Based Testing": f"Transitions between valid and invalid {subject} states are exercised explicitly.",
        "User Behavior Scenarios": f"{subject} under refresh, back navigation, repeated actions, or interrupted flows.",
        "Integration & Dependency Failures": f"{subject} when dependencies fail, timeout, or return inconsistent data.",
        "UI/UX Validation": f"Visibility, disabled states, text accuracy, and responsive behavior in the {subject} flow.",
        "Non-Functional Testing": f"Basic performance, reliability, accessibility, or security expectations for {subject}.",
        "Data Persistence & Consistency": f"{subject} data saving, retention after refresh/login, and backend/UI consistency.",
        "Regression Coverage": f"Targeted regression around existing flows that could break due to {subject}.",
    }
    return hints.get(category, f"Explicit coverage for {category.lower()} in the {subject} workflow.")


def _category_risk_hint(category: str, subject: str) -> str:
    hints = {
        "Happy Path": f"The main {subject} workflow could still fail in the most common user journey.",
        "Negative Scenarios": f"The system may accept invalid {subject} input or expose poor error handling.",
        "Boundary & Edge Cases": f"Limit handling for {subject} may break at thresholds or empty values.",
        "State-Based Testing": f"State transitions in {subject} may allow impossible or unsafe system states.",
        "User Behavior Scenarios": f"Real user interaction patterns may cause duplicate or inconsistent {subject} outcomes.",
        "Integration & Dependency Failures": f"Dependency instability may corrupt or block the {subject} workflow.",
        "UI/UX Validation": f"Users may see incorrect, hidden, or unusable interface states during {subject}.",
        "Non-Functional Testing": f"{subject} may degrade under load or fail baseline reliability/security expectations.",
        "Data Persistence & Consistency": f"{subject} data may not persist or may drift between UI and backend state.",
        "Regression Coverage": f"Previously working behavior may regress while introducing {subject}.",
    }
    return hints.get(category, f"There is still untested risk in {category.lower()} for {subject}.")


def _case_matches_category(test_case: TestCase, category: str) -> bool:
    dimensions = [normalize_dimension_name(value) for value in test_case.coverage_dimensions]
    if category in dimensions:
        return True

    haystack = normalize_text(
        " ".join(
            [
                test_case.category,
                test_case.title,
                test_case.test_type,
                " ".join(test_case.steps),
                test_case.expected_result,
            ]
        )
    )

    if normalize_text(category) in haystack:
        return True

    return any(keyword_matches_text(haystack, keyword) for keyword in CATEGORY_KEYWORDS.get(category, ()))


def _determine_coverage_verdict(report: CoverageReport) -> str:
    if all(item.status == "covered" for item in report.categories):
        return "FULL_COVERAGE"
    return "GAPS_FOUND"


def _build_heuristic_coverage_report(
    context: str,
    impact_analysis: str,
    test_cases: list[TestCase],
) -> CoverageReport:
    subject = infer_subject(context)
    category_reports: list[CoverageCategoryReport] = []
    missing_scenarios: list[str] = []
    untested_risks: list[str] = []
    weak_coverage_areas: list[str] = []

    for category in COVERAGE_CATEGORIES:
        if category == "Regression Coverage" and not impact_analysis:
            category_reports.append(
                CoverageCategoryReport(
                    category=category,
                    status="covered",
                    reason="No impacted historical scope was identified, so targeted regression coverage is not applicable.",
                )
            )
            continue

        evidence_ids = [test_case.case_id for test_case in test_cases if _case_matches_category(test_case, category)]

        if len(evidence_ids) >= 20:
            status = "covered"
            reason = f"{len(evidence_ids)} explicit test case(s) target this dimension, reaching the high-depth target."
        elif len(evidence_ids) > 0:
            status = "partial"
            reason = f"Only {len(evidence_ids)} explicit test case(s) target this dimension, which is below the target depth of 20 for comprehensive coverage."
        else:
            status = "missing"
            reason = "No explicit test case targets this dimension."

        category_missing: list[str] = []
        category_risks: list[str] = []
        if status != "covered":
            category_missing.append(_category_missing_hint(category, subject))
            weak_coverage_areas.append(f"{category}: {reason}")
        if status == "missing":
            category_risks.append(_category_risk_hint(category, subject))

        missing_scenarios.extend(category_missing)
        untested_risks.extend(category_risks)
        category_reports.append(
            CoverageCategoryReport(
                category=category,
                status=status,
                reason=reason,
                evidence_ids=evidence_ids,
                missing_scenarios=category_missing,
                untested_risks=category_risks,
            )
        )

    report = CoverageReport(
        categories=category_reports,
        missing_scenarios=_dedupe_strings(missing_scenarios),
        untested_risks=_dedupe_strings(untested_risks),
        weak_coverage_areas=_dedupe_strings(weak_coverage_areas),
    )
    report = _apply_supplemental_quality_gates(report, test_cases)
    report.coverage_score = calculate_coverage_score(report.categories)
    report.overall_status = _determine_coverage_verdict(report)
    return report


def _normalize_coverage_report_payload(
    payload,
    impact_analysis: str,
    test_cases: list[TestCase],
) -> CoverageReport | None:
    if not isinstance(payload, dict):
        return None

    known_case_ids = {test_case.case_id for test_case in test_cases}
    raw_categories = payload.get("categories") or []
    by_category = {}
    for item in raw_categories:
        if not isinstance(item, dict):
            continue
        category = normalize_dimension_name(str(item.get("category") or "").strip())
        if category in COVERAGE_CATEGORIES:
            by_category[category] = item

    category_reports: list[CoverageCategoryReport] = []
    for category in COVERAGE_CATEGORIES:
        if category == "Regression Coverage" and not impact_analysis:
            category_reports.append(
                CoverageCategoryReport(
                    category=category,
                    status="covered",
                    reason="No impacted historical scope was identified, so targeted regression coverage is not applicable.",
                )
            )
            continue

        item = by_category.get(category, {})
        evidence_ids = [
            str(value).strip()
            for value in item.get("evidence_ids", [])
            if str(value).strip() in known_case_ids
        ]
        status = _normalize_status(str(item.get("status") or "missing"))
        if status == "covered" and len(evidence_ids) < 20:
            status = "partial"
            reason = f"Even though reported as covered, only {len(evidence_ids)} test cases target this category, which is below the target depth of 20 for comprehensive coverage."
        else:
            reason = str(item.get("reason") or "No rationale provided.").strip()

        category_reports.append(
            CoverageCategoryReport(
                category=category,
                status=status,
                reason=reason,
                evidence_ids=evidence_ids,
                missing_scenarios=_split_to_list(item.get("missing_scenarios")),
                untested_risks=_split_to_list(item.get("untested_risks")),
            )
        )

    report = CoverageReport(
        categories=category_reports,
        missing_scenarios=_split_to_list(payload.get("missing_scenarios")),
        untested_risks=_split_to_list(payload.get("untested_risks")),
        weak_coverage_areas=_split_to_list(payload.get("weak_coverage_areas")),
    )
    report.coverage_score = calculate_coverage_score(report.categories)
    report.overall_status = _determine_coverage_verdict(report)
    return report


def _merge_coverage_reports(
    llm_report: CoverageReport | None,
    heuristic_report: CoverageReport,
) -> CoverageReport:
    if not llm_report:
        return heuristic_report

    heuristic_by_category = {item.category: item for item in heuristic_report.categories}
    llm_by_category = {item.category: item for item in llm_report.categories}
    merged_categories: list[CoverageCategoryReport] = []

    for category in COVERAGE_CATEGORIES:
        heuristic_item = heuristic_by_category[category]
        llm_item = llm_by_category.get(category, heuristic_item)

        chosen = heuristic_item if STATUS_RANK[heuristic_item.status] < STATUS_RANK[llm_item.status] else llm_item
        merged_categories.append(
            CoverageCategoryReport(
                category=category,
                status=chosen.status,
                reason=chosen.reason,
                evidence_ids=_dedupe_strings(heuristic_item.evidence_ids + llm_item.evidence_ids),
                missing_scenarios=_dedupe_strings(
                    heuristic_item.missing_scenarios + llm_item.missing_scenarios
                ),
                untested_risks=_dedupe_strings(heuristic_item.untested_risks + llm_item.untested_risks),
            )
        )

    merged_report = CoverageReport(
        categories=merged_categories,
        missing_scenarios=_dedupe_strings(
            heuristic_report.missing_scenarios + llm_report.missing_scenarios
        ),
        untested_risks=_dedupe_strings(
            heuristic_report.untested_risks + llm_report.untested_risks
        ),
        weak_coverage_areas=_dedupe_strings(
            heuristic_report.weak_coverage_areas + llm_report.weak_coverage_areas
        ),
    )
    merged_report.coverage_score = calculate_coverage_score(merged_report.categories)
    merged_report.overall_status = _determine_coverage_verdict(merged_report)
    return merged_report


def _evaluate_coverage(
    context: str,
    impact_analysis: str,
    test_cases: list[TestCase],
) -> CoverageReport:
    heuristic_report = _build_heuristic_coverage_report(context, impact_analysis, test_cases)

    if not _llm_enabled():
        return heuristic_report

    prompt = _build_coverage_evaluation_prompt(context, impact_analysis, test_cases)
    response = _invoke_llm(prompt, "Coverage Evaluation")
    llm_payload = _extract_json_payload(response)
    llm_report = _normalize_coverage_report_payload(llm_payload, impact_analysis, test_cases)
    return _merge_coverage_reports(llm_report, heuristic_report)


def _generate_initial_cases_with_fallback(context: str) -> list[TestCase]:
    seed_cases = parse_test_cases(build_test_cases(context))
    return _decorate_test_cases(seed_cases, "Functional and Non-functional", "Pass 1")


def _make_template_case(
    subject: str,
    title: str,
    preconditions: list[str],
    steps: list[str],
    expected_result: str,
    test_type: str,
    coverage_dimensions: list[str],
    category: str,
    source_pass: str,
) -> TestCase:
    return TestCase(
        title=title,
        preconditions=preconditions,
        steps=steps,
        expected_result=expected_result,
        test_type=test_type,
        category=category,
        source_pass=source_pass,
        coverage_dimensions=coverage_dimensions,
    )


def _generate_gap_cases_with_fallback(
    context: str,
    impact_analysis: str,
    coverage_report: CoverageReport,
    source_pass: str,
) -> list[TestCase]:
    subject = infer_subject(context)
    generated_cases: list[TestCase] = []

    for category_report in coverage_report.categories:
        if category_report.status == "covered":
            continue

        category = category_report.category
        category_label = _preferred_output_category_for_dimension(category)

        templates: list[TestCase] = []
        if category == "State-Based Testing":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify valid {subject} state transition",
                    ["The user has access to the workflow."],
                    [
                        "Open the relevant workflow.",
                        "Move the entity from its initial state to the next valid state.",
                        "Confirm the system records the new status.",
                    ],
                    "The status transition succeeds and the updated state is reflected consistently in the UI and backend.",
                    "State-Based Testing",
                    ["State-Based Testing", "Data Persistence & Consistency"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify invalid {subject} state transition is blocked",
                    ["The workflow is already in a completed or locked state."],
                    [
                        "Attempt an action that is not allowed from the current state.",
                        "Submit the action.",
                    ],
                    "The system blocks the transition and shows a clear state-specific validation message.",
                    "State-Based Testing",
                    ["State-Based Testing", "Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "User Behavior Scenarios":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify repeated user actions do not duplicate {subject}",
                    [],
                    [
                        "Open the workflow.",
                        "Trigger the primary action rapidly or double-click it.",
                        "Observe the resulting system behavior.",
                    ],
                    "The action is processed safely once and duplicate side effects are prevented.",
                    "User Behavior Scenarios",
                    ["User Behavior Scenarios", "Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify {subject} survives refresh or back navigation mid-flow",
                    [],
                    [
                        "Progress partway through the workflow.",
                        "Refresh the page or navigate back and return.",
                        "Continue the workflow.",
                    ],
                    "The workflow state is handled predictably without data corruption or confusing UX.",
                    "User Behavior Scenarios",
                    ["User Behavior Scenarios", "UI/UX Validation"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Integration & Dependency Failures":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify core API health for {subject}",
                    [],
                    [
                        "Call the primary service-health or readiness endpoint that supports the workflow.",
                        "Validate availability and baseline response behavior.",
                    ],
                    "Core API reachability is confirmed and the backend reports a healthy ready state for the workflow.",
                    "Integration & Dependency Failures",
                    ["Integration & Dependency Failures", "Happy Path"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify primary API success contract for {subject}",
                    [],
                    [
                        "Send a valid API request for the primary workflow action.",
                        "Inspect the success status code, response schema, and returned business fields.",
                        "Confirm the backend state change occurred correctly.",
                    ],
                    "The API returns the correct success response, honors its contract, and updates system state consistently.",
                    "Integration & Dependency Failures",
                    ["Happy Path", "Integration & Dependency Failures", "Data Persistence & Consistency"],
                    "API contract, CRUD, integration, and service behavior",
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify graceful handling of dependency timeout during {subject}",
                    [],
                    [
                        "Trigger the workflow while the dependent service responds slowly or times out.",
                        "Observe UI and backend behavior.",
                    ],
                    "The system surfaces a recoverable error, preserves integrity, and avoids partial completion.",
                    "Integration & Dependency Failures",
                    ["Integration & Dependency Failures"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify partial dependency failure does not corrupt {subject} data",
                    [],
                    [
                        "Trigger the workflow when one dependency succeeds and another fails.",
                        "Review persisted data and user-facing status.",
                    ],
                    "The system either rolls back safely or marks the flow consistently without mixed states.",
                    "Integration & Dependency Failures",
                    ["Integration & Dependency Failures", "Data Persistence & Consistency"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Negative Scenarios":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify invalid API payload is rejected for {subject}",
                    [],
                    [
                        "Submit the primary API request with missing or malformed required fields.",
                        "Inspect the returned validation response.",
                    ],
                    "The API rejects the request with the correct validation status and clear error details.",
                    "Negative Scenarios",
                    ["Negative Scenarios", "Integration & Dependency Failures"],
                    "API contract, CRUD, integration, and service behavior",
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify unauthorized API access is blocked for {subject}",
                    [],
                    [
                        "Call the protected API without valid authentication or authorization.",
                        "Inspect the response and downstream effects.",
                    ],
                    "The request is blocked with the correct auth failure response and no unintended state change occurs.",
                    "Negative Scenarios",
                    ["Negative Scenarios", "Integration & Dependency Failures", "Non-Functional Testing"],
                    "API contract, CRUD, integration, and service behavior",
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify invalid {subject} input is rejected safely",
                    [],
                    [
                        "Trigger the workflow with invalid, incomplete, or unauthorized input.",
                        "Submit the action.",
                    ],
                    "The request is rejected safely and the user receives a clear validation or error message.",
                    "Negative Scenarios",
                    ["Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Non-Functional Testing":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify system-wide load handling for {subject}",
                    [],
                    [
                        "Run the primary user-entry and core workflow actions under elevated concurrent load.",
                        "Monitor throughput, error rate, and response stability.",
                    ],
                    "Core platform workflows remain available and behave within acceptable limits under realistic concurrent load.",
                    "Non-Functional Testing",
                    ["Non-Functional Testing"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify dashboard or core-surface SLA for {subject}",
                    [],
                    [
                        "Load the primary dashboard, landing page, or core working surface used for the workflow.",
                        "Measure response time and first-usable render against the target SLA.",
                    ],
                    "The core surface meets the expected response-time SLA under normal operating conditions.",
                    "Non-Functional Testing",
                    ["Non-Functional Testing", "UI/UX Validation"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify long-running endurance and scalability for {subject}",
                    [],
                    [
                        "Sustain realistic traffic and repeated workflow activity for an extended duration.",
                        "Scale the workload upward in stages while monitoring stability and resource behavior.",
                    ],
                    "The platform remains stable over time and scales without unacceptable degradation or inconsistent behavior.",
                    "Non-Functional Testing",
                    ["Non-Functional Testing"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify data leakage and auth-bypass resistance for {subject}",
                    [],
                    [
                        "Attempt to access protected data with insufficient privileges or tampered authentication context.",
                        "Review the response content, status, and resulting system state.",
                    ],
                    "Protected data is not exposed, auth bypass is blocked, and role boundaries remain enforced.",
                    "Non-Functional Testing",
                    ["Non-Functional Testing", "Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Regression Coverage" and impact_analysis:
            templates = [
                _make_template_case(
                    subject,
                    "Verify impacted legacy workflow still behaves as before after the new change",
                    [],
                    [
                        "Execute a historically stable workflow named in the impact analysis.",
                        "Compare the observed behavior with the expected baseline.",
                    ],
                    "The existing workflow continues to work without behavioral regression.",
                    "Regression Coverage",
                    ["Regression Coverage"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    "Verify shared dependencies still support existing consumers after the change",
                    [],
                    [
                        "Exercise an existing feature that shares dependencies with the changed workflow.",
                        "Validate cross-feature behavior and outputs.",
                    ],
                    "Existing features using shared components continue to work correctly.",
                    "Regression Coverage",
                    ["Regression Coverage", "Integration & Dependency Failures"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Data Persistence & Consistency":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify {subject} data persists after save and reload",
                    [],
                    [
                        "Complete the workflow and save the data.",
                        "Refresh or re-open the application.",
                        "Review the saved information.",
                    ],
                    "Saved data is retained correctly after reload and matches the prior submission.",
                    "Data Persistence & Consistency",
                    ["Data Persistence & Consistency"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify DB-level default and schema handling for {subject}",
                    [],
                    [
                        "Create or update the entity with only the allowed minimum required data.",
                        "Inspect the persisted database record or backend representation.",
                    ],
                    "Default values, schema constraints, and stored types are applied correctly at the data layer.",
                    "Data Persistence & Consistency",
                    ["Data Persistence & Consistency"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify backend access control is enforced for {subject} data",
                    [],
                    [
                        "Attempt to read or mutate protected backend data using insufficient permissions or an invalid state.",
                        "Inspect the backend response and stored records.",
                    ],
                    "Unauthorized data operations are blocked at the backend and no improper state change is persisted.",
                    "Data Persistence & Consistency",
                    ["Data Persistence & Consistency", "Negative Scenarios", "Non-Functional Testing"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify UI and backend remain consistent after {subject}",
                    [],
                    [
                        "Complete the workflow.",
                        "Inspect the backend record or API response.",
                        "Compare it with the UI state.",
                    ],
                    "The UI and backend remain synchronized with no data drift.",
                    "Data Persistence & Consistency",
                    ["Data Persistence & Consistency", "Integration & Dependency Failures"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "UI/UX Validation":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify disabled and visible states in the {subject} flow",
                    [],
                    [
                        "Open the workflow in each relevant state.",
                        "Inspect button enablement, visibility, and helper text.",
                    ],
                    "Controls, visibility rules, and copy are accurate for each state.",
                    "UI/UX Validation",
                    ["UI/UX Validation", "State-Based Testing"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify responsive usability for the {subject} workflow",
                    [],
                    [
                        "Open the workflow on desktop and narrow mobile viewports.",
                        "Complete the core interaction.",
                    ],
                    "The layout remains usable and content stays readable across viewports.",
                    "UI/UX Validation",
                    ["UI/UX Validation"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Boundary & Edge Cases":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify minimum boundary handling for {subject}",
                    [],
                    [
                        "Use the smallest allowed input or shortest valid flow variant.",
                        "Submit the workflow.",
                    ],
                    "The system accepts the minimum valid input and behaves correctly.",
                    "Boundary & Edge Cases",
                    ["Boundary & Edge Cases", "Happy Path"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify maximum or empty boundary handling for {subject}",
                    [],
                    [
                        "Use a maximum, empty, or null boundary input depending on the rule.",
                        "Submit the workflow.",
                    ],
                    "The system handles the edge input safely with correct validation or acceptance behavior.",
                    "Boundary & Edge Cases",
                    ["Boundary & Edge Cases", "Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
            ]
        elif category == "Happy Path":
            templates = [
                _make_template_case(
                    subject,
                    f"Verify minimal working system smoke path for {subject}",
                    [],
                    [
                        "Open the application and confirm the primary workflow entry surface is reachable.",
                        "Perform the smallest successful action needed to validate the system is operational.",
                    ],
                    "The system proves basic operational readiness through a minimal successful smoke path.",
                    "Happy Path",
                    ["Happy Path"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify basic navigation smoke coverage for {subject}",
                    [],
                    [
                        "Open the main landing or dashboard surface.",
                        "Navigate to the key page that starts the workflow.",
                        "Confirm the user can reach the primary action without blockers.",
                    ],
                    "Basic navigation works correctly and the user can reach the primary workflow entry point.",
                    "Happy Path",
                    ["Happy Path", "UI/UX Validation"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify the primary successful {subject} journey",
                    [],
                    [
                        "Open the application.",
                        "Complete the workflow using valid data.",
                        "Submit and confirm completion.",
                    ],
                    "The workflow completes successfully and the user sees the correct success outcome.",
                    "Happy Path",
                    ["Happy Path"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify a returning user can repeat the successful {subject} flow",
                    ["A valid account or existing record is available."],
                    [
                        "Re-enter the workflow with valid preconditions.",
                        "Complete the action successfully.",
                    ],
                    "The workflow succeeds consistently for an already known user or entity.",
                    "Happy Path",
                    ["Happy Path", "Regression Coverage" if impact_analysis else "Data Persistence & Consistency"],
                    category_label,
                    source_pass,
                ),
            ]
        else:
            templates = [
                _make_template_case(
                    subject,
                    f"Verify invalid {subject} input is rejected safely",
                    [],
                    [
                        "Trigger the workflow with invalid, incomplete, or unauthorized input.",
                        "Submit the action.",
                    ],
                    "The request is rejected safely and the user receives a clear validation or error message.",
                    "Negative Scenarios",
                    ["Negative Scenarios"],
                    category_label,
                    source_pass,
                ),
                _make_template_case(
                    subject,
                    f"Verify user receives recoverable guidance after a failed {subject} attempt",
                    [],
                    [
                        "Trigger a failure in the workflow.",
                        "Review the error state and retry options.",
                    ],
                    "The system gives actionable feedback and supports safe recovery.",
                    "Negative Scenarios",
                    ["Negative Scenarios", "UI/UX Validation"],
                    category_label,
                    source_pass,
                ),
            ]

        if category_report.status == "missing":
            generated_cases.extend(templates)
        else:
            generated_cases.extend(templates[: min(len(templates), 3)])

    return _decorate_test_cases(generated_cases, "Functional and Non-functional", source_pass)


def _generate_initial_cases(context: str, impact_analysis: str) -> list[TestCase]:
    if not _llm_enabled():
        return _generate_initial_cases_with_fallback(context)

    all_test_cases: list[TestCase] = []
    categories = list(INITIAL_GENERATION_CATEGORIES)
    if impact_analysis:
        categories.append("Dependent and Regression test cases (based on the provided Impact Analysis matrix)")

    print("\nStarting Pass 1 test generation...")
    for category in categories:
        print(f"Generating: {category}...")
        time.sleep(1)
        prompt = _build_generation_prompt(context, category, impact_analysis, all_test_cases)
        generated_cases = _parse_generation_response(
            _invoke_llm(prompt, f"Test Gen: {category}"),
            category,
            "Pass 1",
        )
        generated_cases = _filter_new_cases(generated_cases, all_test_cases)
        assign_test_case_ids(generated_cases, all_test_cases)
        all_test_cases.extend(generated_cases)

        category_cases = [test_case for test_case in all_test_cases if test_case.category == category]
        expansion_prompt = _build_generation_expansion_prompt(
            context,
            category,
            impact_analysis,
            category_cases,
        )
        time.sleep(1)
        expansion_cases = _parse_generation_response(
            _invoke_llm(expansion_prompt, f"Test Gen Expansion: {category}", temperature=0.5),
            category,
            "Pass 1",
        )
        expansion_cases = _filter_new_cases(expansion_cases, all_test_cases)
        assign_test_case_ids(expansion_cases, all_test_cases)
        all_test_cases.extend(expansion_cases)

    if not all_test_cases:
        print("Using local fallback because Pass 1 did not return any parsable test cases.")
        return _generate_initial_cases_with_fallback(context)

    return deduplicate_test_cases(all_test_cases)


def _generate_gap_cases(
    context: str,
    impact_analysis: str,
    existing_cases: list[TestCase],
    coverage_report: CoverageReport,
    source_pass: str,
) -> list[TestCase]:
    if not _llm_enabled():
        gap_cases = _generate_gap_cases_with_fallback(context, impact_analysis, coverage_report, source_pass)
        return _filter_new_cases(gap_cases, existing_cases)

    prompt = _build_gap_generation_prompt(
        context,
        impact_analysis,
        existing_cases,
        coverage_report,
        source_pass,
    )
    generated_cases = _parse_generation_response(
        _invoke_llm(prompt, f"Coverage Gap Fill: {source_pass}", temperature=0.7),
        "Functional and Non-functional",
        source_pass,
    )
    generated_cases = _filter_new_cases(generated_cases, existing_cases)
    if generated_cases:
        return generated_cases

    print("Using local fallback for coverage gap filling because no parsable LLM gap cases were returned.")
    gap_cases = _generate_gap_cases_with_fallback(context, impact_analysis, coverage_report, source_pass)
    return _filter_new_cases(gap_cases, existing_cases)


def _extract_initial_gaps(coverage_report: CoverageReport) -> list[str]:
    gaps = [
        f"{item.category}: {item.reason}"
        for item in coverage_report.categories
        if item.status != "covered"
    ]
    gaps.extend(coverage_report.weak_coverage_areas)
    return _dedupe_strings(gaps)

INITIAL_GENERATION_CATEGORIES = [
    "Foundational Smoke and Sanity",
    "End-to-End Business Workflows",
    "Core Functional Requirements",
    "Detailed Functional Rules and Constraints",
    "Non-Functional Expectations",
    "Front-end UI and UX",
    "Front-end Responsive and Cross-Browser",
    "Back-end Database Persistence and Integrity",
    "Back-end Schema and Audit Validation",
    "API Contract and Schema Validation",
    "API CRUD and State Change Correctness",
    "Positive Happy Path Scenarios",
    "Negative Error Handling and Validation Scenarios",
    "State Transitions and Lifecycle flows",
    "User Behavior",
    "Concurrency and Race Conditions",
    "Interruption and Recovery scenarios",
    "Boundary and Edge Cases",
    "System Performance and Latency SLA",
    "Security, Auth, and Role-Based Access Control",
]

MAX_ITERATIONS = 10
STATUS_RANK = {
    "missing": 0,
    "partial": 1,
    "covered": 2,
}


@lru_cache(maxsize=1)
def _get_llm(temperature: float = 0):
    """Initializes and returns the LangChain Azure OpenAI Chat model."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise ImportError("Please install the 'langchain-openai' package to use Azure OpenAI models.") from exc

    return AzureChatOpenAI(
        azure_deployment=get_chat_model_name(),
        openai_api_version=get_azure_api_version(),
        azure_endpoint=get_azure_endpoint(),
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
        temperature=temperature,
        max_retries=0,
        timeout=300,
    )


def _legacy_generate_test_cases(context: str, impact_analysis: str = "") -> str:
    """
    Acts as a 'Senior QA Engineer'.
    Instead of asking for all tests at once (which usually hits output length limits),
    this function loops through different testing domains to get exhaustive results.
    
    Args:
        context: The structural requirements text (Features, Rules, etc.) from the Analyzer.
        impact_analysis: (Optional) The regression report listing which modules were affected.
    """
    
    # If no API key is set, redirect to the local 'dummy' test generator.
    if not has_valid_openai_key():
        return build_test_cases(context)

    # Define the core "pillars" of testing. Each string will trigger a separate LLM call.
    categories = [
        "Foundational smoke, sanity, and end-to-end business workflow",
        "Functional and Non-functional",
        "Front-end UI",
        "Back-end database and schema validation",
        "API",
        "Positive and negative scenarios",
        "State transitions, user behavior, concurrency, and interruption scenarios",
        "Edge cases",
        "Performance-related",
    ]

    # If the Impact Analysis step found historical risks, add a dedicated regression test phase.
    if impact_analysis:
        categories.append(
            "Dependent and Regression test cases (based on the provided Impact Analysis matrix)"
        )

    # List to store the text results from each category iteration.
    all_test_cases = []
    
    print("\nStarting iterative test case generation for comprehensive coverage...")

    # Loop through each category sequentially to ensure deep focus on every testing area.
    for category in categories:
        print(f"Generating: {category}...")
        
        # Build a highly specific prompt for the current testing category.
        prompt = f"""
        You are a Senior QA Engineer working in a top-tier product company (Google-level quality expectations).

        Your task is NOT just to generate test cases, but to DESIGN a COMPLETE and SYSTEMATIC test suite using structured QA thinking.

        ========================
        MANDATORY TEST DESIGN STRATEGY
        ========================

        You MUST generate test cases by systematically covering ALL of the following dimensions:

        1. Happy Path Scenarios
        - Validate correct and expected system behavior.

        2. Negative Scenarios
        - Invalid inputs
        - Incorrect user actions
        - System misuse

        3. Boundary & Edge Cases
        - Min/max values
        - Limits and thresholds
        - Empty/null inputs
        - Just below / above limits

        4. State-Based Testing (CRITICAL)
        - Identify all possible system states
        - Validate transitions between states
        - Include invalid state transitions

        5. User Behavior Scenarios
        - Rapid clicks, double actions
        - Refresh, back navigation
        - Multi-tab usage
        - Interrupted flows

        6. Integration & Dependency Testing
        - API failures
        - Delayed responses
        - Partial system failures
        - Data inconsistencies

        7. UI/UX Validation
        - Disabled states
        - Visibility conditions
        - Text/content correctness
        - Cross-page consistency

        8. Non-Functional Scenarios
        - Performance (basic level)
        - Security (basic validation)
        - Reliability

        9. Data & Persistence Validation
        - Data saved correctly
        - State retained after refresh/login
        - Backend vs UI consistency

        10. Regression / Dependency Testing (IMPORTANT)
        - If category is "Dependent and Regression test cases":
          - STRICTLY use Impact Analysis Matrix
          - Generate ONLY targeted regression cases
          - Focus on what can BREAK in existing functionality
        - If NO_IMPACT → DO NOT generate regression cases

        ========================
        INTERNAL THINKING STEP (DO NOT SKIP)
        ========================

        First, internally identify:
        - All possible system states
        - State transitions
        - High-risk areas
        - Failure points

        DO NOT output this analysis.
        Use it internally to improve test case quality and coverage.

        ========================
        STRICT QUALITY RULES
        ========================

        - DO NOT generate generic or shallow test cases
        - EACH test case must validate a UNIQUE behavior or risk
        - AVOID duplication
        - Ensure FULL coverage of the feature
        - Think like a user + system + attacker

        ========================
        OUTPUT FORMAT (STRICT)
        ========================

        Test Case ID: [Category Prefix]-001
        Title:
        Preconditions:
        Steps:
        Expected Result:
        Test Type: (Happy / Negative / Boundary / State / Integration / UI / Regression / etc.)

        ========================
        INPUT CONTEXT
        ========================

        Category: **{category}**

        Current Requirement Analysis:
        {context}
        """

        # If we are in the regression cycle, inject the impact matrix to guide the AI.
        if impact_analysis and "Regression" in category:
            prompt += f"\nImpact Analysis Matrix:\n{impact_analysis}\n"

        try:
            # Capture the token usage for each testing category as we iterate through the loop.
            with get_openai_callback() as cb:
                # Call the LLM and get its response content.
                result = _get_llm().invoke(prompt).content
                
                # Record the usage findings with a dynamic category label to identify exactly what was generated.
                usage_tracker.record_usage(f"Test Gen: {category}", cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)
                
            # Wrap the response in a category header for the final report.
            all_test_cases.append(f"### Category: {category}\n\n{result}\n")
        except Exception as exc:
            # Handle API/Network errors for this specific category and continue to the next one.
            print(f"Failed to generate {category}: {type(exc).__name__}")

    # If every category iteration failed, trigger the local fallback as a last resort.
    if not all_test_cases:
        print("Using local test generation fallback due to zero categories generated.")
        return build_test_cases(context)

    # Join all separate category results into one massive, master test suite string.
    return "\n\n".join(all_test_cases)


def generate_test_cases(context: str, impact_analysis: str = "") -> str:
    """
    Generates, self-evaluates, and refines the test suite until coverage is acceptable.
    """
    all_test_cases = _generate_initial_cases(context, impact_analysis)
    all_test_cases = deduplicate_test_cases(all_test_cases)
    assign_test_case_ids(all_test_cases)

    additional_test_cases: list[TestCase] = []
    final_coverage_report: CoverageReport | None = None
    initial_gaps: list[str] = []

    print("\nStarting Pass 2 self-evaluation...")
    for iteration in range(1, MAX_ITERATIONS + 1):
        coverage_report = _evaluate_coverage(context, impact_analysis, all_test_cases)
        coverage_report.iterations_used = iteration
        final_coverage_report = coverage_report

        print(
            f"Coverage iteration {iteration}: "
            f"score={coverage_report.coverage_score}% verdict={coverage_report.overall_status}"
        )

        if iteration == 1:
            initial_gaps = _extract_initial_gaps(coverage_report)

        if coverage_report.overall_status == "FULL_COVERAGE":
            break

        if iteration == MAX_ITERATIONS:
            break

        time.sleep(1)
        source_pass = f"Pass 2 - Iteration {iteration}"
        gap_cases = _generate_gap_cases(
            context,
            impact_analysis,
            all_test_cases,
            coverage_report,
            source_pass,
        )

        if not gap_cases:
            print("No additional gap-filling cases were generated in this iteration.")
            break

        assign_test_case_ids(gap_cases, all_test_cases)
        additional_test_cases.extend(gap_cases)
        all_test_cases.extend(gap_cases)
        all_test_cases = deduplicate_test_cases(all_test_cases)

    if not final_coverage_report:
        final_coverage_report = _build_heuristic_coverage_report(context, impact_analysis, all_test_cases)

    all_test_cases = _sort_test_cases_for_output(deduplicate_test_cases(all_test_cases))
    additional_test_cases = _sort_test_cases_for_output(deduplicate_test_cases(additional_test_cases))

    result = TestSuiteResult(
        final_test_suite=all_test_cases,
        coverage_report=final_coverage_report,
        initial_gaps=initial_gaps,
        additional_test_cases=additional_test_cases,
    )
    return render_final_report(result)
