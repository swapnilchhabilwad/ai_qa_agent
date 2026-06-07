import os  # Used for interacting with the operating system environment variables.
from functools import lru_cache  # Used for caching the environment loading result to ensure speed and consistency.

# Import load_dotenv from the python-dotenv library.
# This library reads key-value pairs from a .env file and sets them as environment variables.
from dotenv import load_dotenv


# List of common strings used by users when they haven't yet filled in their real API keys.
# We use this to detect if the framework is 'misconfigured' and should use a local fallback or error out.
PLACEHOLDER_MARKERS = (
    "your_openai_key",
    "your_openai_api_key",
    "your_api_key",
    "your_azure_endpoint",
    "your_azure_deployment",
    "replace_me",
    "example",
)


@lru_cache(maxsize=1)  # Ensures load_dotenv is only physically executed once during the application lifecycle.
def load_environment() -> bool:
    """
    Triggers the loading of the .env file globally.
    Returns True upon successful load attempt.
    """
    load_dotenv()
    return True


def _looks_like_placeholder(value: str) -> bool:
    """
    Internal helper to check if a string is a common placeholder rather than a real key.
    """
    normalized = value.strip().lower()

    # If the value is empty, treat it as a missing/placeholder value.
    if not normalized:
        return True

    # Check if for example the user left "your_api_key_here" in the .env.
    if normalized.startswith("your_"):
        return True

    # Check against the list of known dummy strings.
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def has_valid_env(name: str) -> bool:
    """
    Boolean check to see if an environment variable exists AND doesn't look like a placeholder.
    """
    load_environment()  # Ensure vars are loaded.
    value = os.getenv(name, "").strip()
    return bool(value) and not _looks_like_placeholder(value)


def has_valid_openai_key() -> bool:
    """
    Specific check for the Azure OpenAI API key.
    We maintain this name for backward compatibility with older 'openai' generic checks.
    """
    return has_valid_env("AZURE_OPENAI_API_KEY")


def get_required_env(name: str) -> str:
    """
    Fetches an environment variable but crashes with a helpful error if it is missing or fake.
    This prevents the framework from running with empty configurations which causes cryptic errors.
    """
    load_environment()
    value = os.getenv(name, "").strip()

    # If the key is entirely missing from the .env or shell environment.
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Add it to .env or export it in your shell."
        )

    # If the key was found but still looks like 'your_api_key'.
    if _looks_like_placeholder(value):
        raise RuntimeError(
            f"Environment variable '{name}' still contains a placeholder value. "
            "Update .env with a real credential before running the framework."
        )

    return value


def get_chat_model_name() -> str:
    """Gets the Azure Deployment Name for the Chat Model (e.g., gpt-4o)."""
    load_environment()
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "").strip()


def get_embedding_model_name() -> str:
    """Gets the Azure Deployment Name for the Embeddings Model (e.g., text-embedding-3-small)."""
    load_environment()
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "").strip()


def get_azure_endpoint() -> str:
    """Returns the base URL for the Azure OpenAI resource."""
    return get_required_env("AZURE_OPENAI_ENDPOINT")


def get_azure_api_version() -> str:
    """
    Returns the target API version string. 
    Defaults to '2024-02-15-preview' if not specified in .env.
    """
    load_environment()
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()


def get_daily_token_limit() -> int:
    """
    Fetches the daily token limit from .env.
    Defaults to 100,000 if not specified.
    """
    load_environment()
    limit = os.getenv("AZURE_OPENAI_TOKEN_LIMIT_DAILY", "100000").strip()
    try:
        return int(limit)
    except ValueError:
        return 100000


def get_pricing_rates() -> dict:
    """
    Returns the cost-per-token rates for the current models.
    Values are stored as USD per 1 token.
    """
    load_environment()
    
    # Get rates from .env or use defaults for gpt-4o-mini and ada-002.
    # We divide by 1,000,000 because rates are typically quoted per 1M tokens.
    prompt_rate = float(os.getenv("COST_PER_1M_PROMPT", "0.15")) / 1_000_000
    completion_rate = float(os.getenv("COST_PER_1M_COMPLETION", "0.60")) / 1_000_000
    embedding_rate = float(os.getenv("COST_PER_1M_EMBEDDING", "0.10")) / 1_000_000
    
    return {
        "prompt": prompt_rate,
        "completion": completion_rate,
        "embedding": embedding_rate
    }
