import csv       # Used to read back the CSV file written by export_to_csv() so we can assert on its content.
import io        # Provides StringIO — an in-memory file object that behaves like a real file, used to avoid touching the disk in tests.
import unittest  # Python's built-in testing framework — provides TestCase, assertions, and test discovery.
from unittest.mock import patch  # 'patch' lets us replace real functions/objects with mocks during a test.

from utils.exporter import export_to_csv  # The function being tested — converts markdown test output into a CSV file.


# A minimal but realistic markdown test report used as the input for the test below.
# It contains exactly 2 test cases in the format the exporter expects to parse.
SAMPLE_REPORT = """
## Final Test Suite

### Category: State transitions, user behavior, concurrency, and interruption scenarios

Test Case ID: STA-001
Title: Verify valid status transition
Steps:
1. Open the workflow.
2. Move the entity to the next valid state.
Expected Result: The new status is saved correctly.
Test Type: State-Based Testing
Generation Source: Pass 2 - Iteration 1
Coverage Dimensions: State-Based Testing, Data Persistence & Consistency

### Category: State transitions, user behavior, concurrency, and interruption scenarios

Test Case ID: USR-001
Title: Verify repeated clicks do not duplicate submission
Steps:
1. Open the workflow.
2. Click the primary action rapidly.
Expected Result: Only one submission is processed.
Test Type: User Behavior Scenarios
Generation Source: Pass 2 - Iteration 1
Coverage Dimensions: User Behavior Scenarios
"""


class NonClosingStringIO(io.StringIO):
    """
    A subclass of StringIO that overrides the close() method to be a no-op.
    This is needed because the exporter opens a file using a 'with' statement,
    which calls close() on exit. If close() were called on a real StringIO,
    its content would be lost before we could assert on it.
    By making close() do nothing, we can read the buffer after the 'with' block.
    """
    def close(self):
        pass  # Intentionally do nothing — prevents the in-memory buffer from being wiped prematurely.


class ExporterTests(unittest.TestCase):
    """Groups all unit tests for the export_to_csv() function."""

    def test_export_to_csv_writes_structured_rows(self):
        """
        Verifies that export_to_csv() correctly:
        1. Parses the markdown report into TestCase objects.
        2. Writes a header row and one data row per test case to the CSV.
        3. Assigns the correct values to specific columns like 'Test Case ID',
           'Generation Source', and 'Coverage Dimensions'.
        """
        buffer = NonClosingStringIO()  # Create an in-memory buffer to capture file output without touching the disk.

        with (
            patch("builtins.open", return_value=buffer),  # Redirect all open() calls to our in-memory buffer.
            patch("os.makedirs"),                          # Prevent actual directory creation during the test.
            patch("builtins.print"),                       # Suppress console output so test results stay clean.
        ):
            export_to_csv(SAMPLE_REPORT, "mocked_output/report.csv")  # Run the function under test.

        # Read back the content written to the in-memory buffer and parse it as CSV.
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))

        # Assert exactly 2 rows were written — one per test case in SAMPLE_REPORT.
        self.assertEqual(len(rows), 2)

        # Assert the first row contains the correct test case ID from the first test case.
        self.assertEqual(rows[0]["Test Case ID"], "STA-001")

        # Assert the first row has the correct Generation Source value.
        self.assertEqual(rows[0]["Generation Source"], "Pass 2 - Iteration 1")

        # Assert the second row's coverage dimensions were correctly written.
        self.assertEqual(rows[1]["Coverage Dimensions"], "User Behavior Scenarios")


if __name__ == "__main__":
    # Allows this test file to be run directly with 'python test_exporter.py'
    # in addition to being discovered by pytest or unittest discovery.
    unittest.main()
