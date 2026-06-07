import os   # Used to construct the absolute path to the project root for sys.path manipulation.
import sys  # Used to add the project root to Python's module search path so local imports work when running this file directly.

# Add the project root directory to sys.path so Python can find 'utils' and 'agents' packages
# when this script is run directly from any working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_environment  # Import the environment loader so the .env file (API keys etc.) is parsed before making any LLM calls.
from agents.test_generator import _get_llm  # Import the internal LLM factory function directly to test the connection in isolation.

load_environment()  # Trigger loading of the .env file so AZURE_OPENAI_API_KEY and other secrets are available as environment variables.
llm = _get_llm()   # Initialize the LLM connection object — this verifies that the credentials and endpoint are working.

# Define a minimal test prompt to send directly to the LLM.
# The purpose is to confirm the model can receive a prompt and return a valid structured response
# without going through the full analysis + test generation pipeline.
prompt = """
You are a Senior QA Engineer.
Return plain MARKDOWN only. Do not use JSON.
Use exactly this markdown format for each test case:

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

Required Output Categories: Functional
Please respond now.
"""

print("Sending request...")  # Inform the developer that the API call is about to be made.
response = llm.invoke(prompt)  # Send the prompt to the LLM and block until a response is returned.
print("Content:", repr(response.content))  # Print the raw text content of the response (repr helps show any hidden whitespace or escape characters).
print("Response Metadata:", response.response_metadata)  # Print extra metadata (e.g., model name, finish reason, token counts) returned by the API alongside the content.
