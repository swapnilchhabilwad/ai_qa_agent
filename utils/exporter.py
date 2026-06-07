import csv  # Built-in library to handle CSV creation and writing.
import os   # Used for path manipulation and directory creation.

# Import the deduplication and parsing utilities from the shared test_suite module.
# - parse_test_cases: Converts raw markdown output from the LLM into structured TestCase objects.
# - deduplicate_test_cases: Removes duplicate test cases before writing to avoid redundant rows in the CSV.
from utils.test_suite import deduplicate_test_cases, parse_test_cases


def export_to_csv(markdown_content: str, output_path: str):
    """
    Parses the generated test cases from markdown text and exports them to a CSV file.
    This is the final step of the pipeline — it converts the AI's markdown output into a
    spreadsheet format that can be imported into any test management tool (e.g., Jira, TestRail).
    """
    # Parse the markdown string into a list of TestCase objects, then remove any duplicates.
    # Deduplication is important here because multiple generation passes may produce overlapping cases.
    parsed_cases = deduplicate_test_cases(parse_test_cases(markdown_content))

    # If parsing returned nothing (e.g., malformed LLM output), abort early and warn the user.
    if not parsed_cases:
        print("Warning: No test cases could be parsed from the output. Export cancelled.")
        return

    # Ensure the full directory path exists before opening the file for writing.
    # 'exist_ok=True' prevents an error if the directory already exists.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Define the exact column names for the output CSV file.
    # These match the fields on the TestCase dataclass so each row is a complete test case record.
    keys = [
        "Category",           # The testing category (e.g., "Positive Happy Path Scenarios").
        "Test Case ID",       # The unique identifier generated for the test case (e.g., "PHS-001").
        "Title",              # A short, descriptive name for what the test validates.
        "Preconditions",      # Any setup steps needed before the test can run.
        "Steps",              # The exact actions to perform during execution.
        "Expected Result",    # What the system should do if the test passes.
        "Test Type",          # The testing dimension (e.g., "Functional", "Performance").
        "Generation Source",  # Which pass of generation produced this test case (e.g., "Pass 1").
        "Coverage Dimensions",# The coverage labels this test case contributes to.
    ]

    try:
        # Open the output file for writing. 'newline=""' prevents Python from adding extra blank lines on Windows.
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Create a DictWriter which maps dictionary keys to CSV columns automatically.
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()  # Write the column header row at the top of the file.

            # Iterate through every parsed test case and write it as a single row.
            for test_case in parsed_cases:
                writer.writerow(
                    {
                        "Category": test_case.category,  # The output category bucket this test belongs to.
                        "Test Case ID": test_case.case_id,  # The unique ID assigned during ID-assignment step.
                        "Title": test_case.title,  # Human-readable description of the scenario.
                        "Preconditions": "\n".join(test_case.preconditions).strip(),  # Join list items with newlines for multi-line cell display.
                        "Steps": "\n".join(test_case.steps).strip(),  # Join each step with a newline to keep them readable inside one cell.
                        "Expected Result": test_case.expected_result.strip(),  # Remove any leading/trailing whitespace from the result text.
                        "Test Type": test_case.test_type,  # The dimension label (e.g., "Happy Path", "Security").
                        "Generation Source": test_case.source_pass,  # Tracks which iteration of the pipeline created this case.
                        "Coverage Dimensions": ", ".join(test_case.coverage_dimensions),  # Comma-separated list of coverage labels.
                    }
                )
        print(f"\n[Export Success] Successfully exported {len(parsed_cases)} test cases to: {output_path}")
    except Exception as e:
        # Catch any I/O or permission error and report it gracefully without crashing the full pipeline.
        print(f"\n[Export Error] Failed to write CSV file: {e}")
