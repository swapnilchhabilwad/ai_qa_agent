from fastapi import FastAPI, HTTPException  # FastAPI is the web framework; HTTPException lets us send HTTP error responses.
from pydantic import BaseModel  # BaseModel is used to define strongly-typed request/response schemas for the API.
import os   # Used for file path operations such as checking existence and joining directory parts.
import sys  # Provides access to system-level functions (imported for potential future use in this module).
from main import run_agent, load_environment, export_to_csv  # Import core orchestration, environment loading, and CSV export from the CLI entry point.
from utils.usage_tracker import usage_tracker  # Import the shared token-usage tracker to expose billing data in API responses.

# Load environment variables (API Keys, config) so they are available before the app starts serving requests.
load_environment()

# Create the FastAPI application instance.
# The 'title' appears in the auto-generated API documentation at /docs.
app = FastAPI(title="AI QA Agent API")


class TestGenerationRequest(BaseModel):
    """
    Defines the expected JSON body for the /generate-tests endpoint.
    Pydantic automatically validates that each incoming request contains the correct types.
    """
    file_path: str          # Full path to the PRD document the user wants to generate tests for.
    project: str = "Greenprint"  # Project name used to scope memory and output directories; defaults to 'Greenprint'.
    query: str = "Generate comprehensive test cases exhaustively for all functional capabilities"  # The instruction passed to the AI; defaults to a broad coverage query.


def save_results_logic(result: str, file_path: str, project_name: str = "default") -> str:
    """
    Duplicate of result-saving logic to keep it within app.py context if needed,
    though importing from main.py is preferred if refactored.
    """
    # Extract the base filename (excluding extension) to name the output file.
    # e.g., "prd1.txt" becomes "prd1" so the output becomes "prd1_test_cases.csv".
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_filename = f"{base_name}_test_cases.csv"  # Append a suffix so it's clear this is an output file.

    # Determine Module from file_path by looking at the folder structure.
    # Expected convention: data/Projects/<ProjectName>/<ModuleName>/prd.txt
    path_parts = os.path.normpath(file_path).split(os.sep)  # Normalize slashes and split into path segments.
    module_name = ""  # Will hold the sub-module name if one is found in the path hierarchy.
    if "Projects" in path_parts:  # Only proceed if the path follows the expected 'Projects' convention.
        idx = path_parts.index("Projects")  # Find the index of the 'Projects' folder in the path list.
        if len(path_parts) > idx + 2:  # Make sure there are at least 2 parts after 'Projects' (project name + module).
            module_name = path_parts[idx + 2]  # The part two levels after 'Projects' is the module name (e.g., 'Onboarding').
            if module_name == os.path.basename(file_path):  # Guard: if it's the filename itself, there is no module subfolder.
                module_name = ""

    # Define the output directory path for the results, starting at 'data/Results/'.
    results_dir = os.path.join("data", "Results")
    if project_name != "default":  # Add a project-specific subdirectory unless it's the generic 'default' scope.
        results_dir = os.path.join(results_dir, project_name)

    if module_name:  # If a module sub-folder was found, nest the results one level deeper inside it.
        results_dir = os.path.join(results_dir, module_name)

    if not os.path.exists(results_dir):  # Only create the directory if it doesn't already exist.
        os.makedirs(results_dir)  # Create all intermediate directories in the path if needed.

    csv_path = os.path.join(results_dir, csv_filename)  # Combine the directory and filename into a full output path.
    export_to_csv(result, csv_path)  # Delegate the actual CSV writing to the shared exporter utility.
    return csv_path  # Return the final path so it can be included in the API response.


@app.post("/generate-tests")  # Register this function as the handler for POST requests to /generate-tests.
async def generate_tests(request: TestGenerationRequest):
    """
    The primary API endpoint.
    Accepts a PRD file path and project name, runs the full AI test generation pipeline,
    saves the result to CSV, and returns a structured response with test cases and usage stats.
    """
    # Validate that the PRD file actually exists on the server before starting the expensive pipeline.
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {request.file_path}")  # Return HTTP 404 if the file is missing.

    try:
        # Run the agent logic — this is the same core flow triggered by the CLI entry point in main.py.
        result = run_agent(request.query, request.file_path, project_name=request.project)

        # Save results to CSV using the path-construction logic defined above.
        csv_path = save_results_logic(result, request.file_path, request.project)

        # Retrieve the current session's token usage and cost for inclusion in the response.
        usage = usage_tracker.get_session_summary()

        # Return a structured JSON response with the status, output path, content, and billing data.
        return {
            "status": "success",       # Indicates the pipeline completed without errors.
            "csv_path": csv_path,      # Path where the test cases have been saved.
            "test_cases": result,      # The full markdown test case output.
            "usage": usage             # Token and cost breakdown for this session.
        }
    except Exception as e:
        # Catch any unexpected error and convert it to an HTTP 500 Internal Server Error.
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage")  # Register this function as the handler for GET requests to /usage.
async def get_usage():
    """
    A simple diagnostic endpoint that returns the current session's token usage and cost summary.
    Useful for monitoring consumption without triggering the full test generation pipeline.
    """
    return usage_tracker.get_session_summary()  # Return the live session usage dictionary.


if __name__ == "__main__":
    # This block only runs when the file is executed directly (e.g., python app.py),
    # not when it is imported as a module by another process.
    import uvicorn  # uvicorn is the ASGI server that serves the FastAPI application.
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start serving on all network interfaces at port 8000.
