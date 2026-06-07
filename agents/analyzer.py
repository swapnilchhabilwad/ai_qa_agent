from functools import lru_cache  # Used for caching function results to avoid redundant expensive initializations.

# Import configuration utilities to access environment variables and model settings.
from utils.config import (
    get_chat_model_name,      # Helper to get the Azure OpenAI deployment name.
    get_required_env,          # Helper to fetch vital environment variables (e.g., API Keys).
    has_valid_openai_key,      # Helper to check if OpenAI/Azure setup is active.
    get_azure_endpoint,        # Helper to get the Azure API endpoint URL.
    get_azure_api_version      # Helper to get the specific Azure API version string.
)
# Import a local fallback mechanism for analysis when API keys are missing.
from utils.local_fallback import build_analysis
# Import the LangChain callback manager to capture token usage in real-time.
from langchain_community.callbacks import get_openai_callback
# Import the global usage tracker to persist token consumption to the logs.
from utils.usage_tracker import usage_tracker

@lru_cache(maxsize=1)  # Ensures we only initialize the LLM object once and reuse it (Singleton pattern).
def _get_llm():
    """
    Initializes and returns the Langchain Azure OpenAI Chat Model.
    This function handles the specific connection details for Azure OpenAI services.
    """
    try:
        # Import the AzureChatOpenAI class from the langchain-openai package.
        from langchain_openai import AzureChatOpenAI
    except ImportError:
        # If the required library is not installed, prompt the user with instructions.
        raise ImportError("Please install the 'langchain-openai' package to use Azure OpenAI models.")

    # Configure and return the chat model instance.
    return AzureChatOpenAI(
        azure_deployment=get_chat_model_name(),  # The specific deployment name in Azure.
        openai_api_version=get_azure_api_version(),  # The API version for compatibility.
        azure_endpoint=get_azure_endpoint(),         # The regional Azure resource endpoint.
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),  # The authentication key.
        temperature=0,  # Set to 0 for deterministic, consistent outputs (essential for technical analysis).
        max_retries=0,
        timeout=20,
    )


def analyze_requirement(context: str) -> str:
    """
    Acts as an 'AI Business Analyst'.
    It parses raw document text and transforms it into structured requirement categories.
    
    Args:
        context: The raw massive block of text extracted from the source PRD document.
    """
    # If no API key is available, use a simple local text-processing function as a fallback.
    if not has_valid_openai_key():
        return build_analysis(context)

    # Construct the instruction (System Prompt) for the AI, injecting the raw text context.
    prompt = f"""
    You are a QA Business Analyst.

    Analyze the following requirement content and extract:

    1. Features: Main functionalities of the application.
    2. User flows: Step-by-step paths users follow to complete tasks.
    3. Business rules: Logic and constraints governing the system behavior.
    4. Edge conditions: Unusual or extreme scenarios that should be tested.

    Content:
    {context}
    """

    try:
        # Start a context manager to intercept and record token consumption for this specific call.
        with get_openai_callback() as cb:
            # Send the prompt to the LLM and return the string content of its response.
            response = _get_llm().invoke(prompt).content
            
            # Record the usage (prompt, completion, and total tokens) with a descriptive label for the audit log.
            usage_tracker.record_usage("Requirement Analysis", cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)
            
            return response
    except Exception as exc:
        # If the API call fails (e.g., network issues), log the error and use the local fallback.
        print(
            f"OpenAI analysis unavailable ({type(exc).__name__}). "
            "Using local analysis fallback."
        )
        return build_analysis(context)


def analyze_impact(new_analysis: str, historical_context: str) -> str:
    """
    Acts as a 'Senior QA Architect'.
    It compares new requirements against old system memory to detect functional regressions.
    
    Args:
        new_analysis: The features extracted from the PRD currently being processed.
        historical_context: Merged features from previously analyzed PRDs (stored in memory).
    """
    # Defensive check: if there is no historical data, there is nothing for the new feature to impact.
    if not historical_context:
        return ""
        
    # Standard check for API key availability before attempting LLM calls.
    if not has_valid_openai_key():
        return ""

    # Construct a specialized prompt for regression and impact assessment.
    prompt = f"""
    You are a Senior QA Architect. 
    
    You are reviewing a newly requested Feature Analysis against previously existing historical System Features.
    
    Your goal is to determine if the NEW Feature Analysis impacts or breaks any of the functionality specified in the Historical System Features.
    
    If it DOES impact previous features, return a summary listing exactly WHICH features are impacted and how.
    If it DOES NOT impact previous features at all, DO NOT return any text related to regression testing. Only return the exact phrase: "NO_IMPACT".
    
    New Feature Analysis:
    {new_analysis}
    
    Historical System Features:
    {historical_context}
    """

    try:
        # Use the callback manager to capture precise token usage for the impact assessment.
        with get_openai_callback() as cb:
            # Invoke the LLM to get the impact report.
            result = _get_llm().invoke(prompt).content
            
            # Persist the token usage count with a descriptive task label for managerial reporting.
            usage_tracker.record_usage("Impact Analysis", cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)
        
        # Check if the AI explicitly stated there is no impact.
        # The length check prevents false negatives if "NO_IMPACT" is mentioned within a larger explanation.
        if "NO_IMPACT" in result and len(result) < 20:
            return ""
            
        return result  # Return the detailed impact summary if found.
    except Exception as exc:
        # Log failure and return an empty string (assume no impact rather than crashing).
        print(f"Impact Analysis failed ({type(exc).__name__}). Passing control blindly.")
        return ""
