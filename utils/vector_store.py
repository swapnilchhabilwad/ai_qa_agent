import json  # Used for saving and loading manifest/local-store JSON files.
import os    # Used for path operations and directory creation.
from collections import Counter  # Used for simple frequency-based ranking in the local search fallback.
from dataclasses import dataclass  # Used to create lightweight, structured document objects.
from functools import lru_cache  # Used to cache the expensive embeddings model initialization.

# Import configuration utilities to access Azure/OpenAI settings.
from utils.config import (
    get_embedding_model_name, # Helper to get the embedding model ID from environment.
    get_required_env,          # Helper to fetch vital secrets.
    has_valid_openai_key,      # Helper to check if we should use AI or Local mode.
    get_azure_endpoint,        # Helper for Azure API URL.
    get_azure_api_version      # Helper for Azure API versioning.
)
# Import the tokenize helper to process search queries locally.
from utils.local_fallback import tokenize


# Define the relative paths where the persistence artifacts are stored.
EMBEDDING_PATH = "embeddings"
# FAISS binary index file.
INDEX_FILE = os.path.join(EMBEDDING_PATH, "index.faiss")
# FAISS metadata pickle file.
PICKLE_FILE = os.path.join(EMBEDDING_PATH, "index.pkl")
# Metadata manifest to check if the store is up-to-date with the source PRD.
MANIFEST_FILE = os.path.join(EMBEDDING_PATH, "manifest.json")
# JSON file used to store raw text chunks when running in offline/local mode.
LOCAL_STORE_FILE = os.path.join(EMBEDDING_PATH, "local_store.json")


@dataclass
class LocalDocument:
    """A lightweight mock of the Langchain Document class used for local fallback search."""
    page_content: str


class LocalVectorStore:
    """
    A simple, keyword-overlap based search engine.
    Used when FAISS/OpenAI are unavailable.
    """
    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def similarity_search(self, query: str, k: int = 3) -> list[LocalDocument]:
        """
        Ranks chunks based on keyword overlap with the search query.
        """
        # Count frequency of words in the query.
        query_counts = Counter(tokenize(query))
        ranked: list[tuple[float, str]] = []

        for chunk in self.chunks:
            # Count frequency of words in the current PRD chunk.
            chunk_counts = Counter(tokenize(chunk))
            # Calculate how many query words also appear in this chunk.
            overlap = sum(
                min(query_counts[token], chunk_counts[token]) for token in query_counts
            )
            # Calculate "quality" scores based on overlap, coverage, and density.
            coverage = overlap / max(sum(query_counts.values()), 1)
            density = overlap / max(sum(chunk_counts.values()), 1)
            # Weighted score: Give more weight to coverage (meaning finding more query terms).
            score = (coverage * 0.7) + (density * 0.3)
            ranked.append((score, chunk))

        # Sort chunks by score in descending order.
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in ranked[:k]]

        # If no chunks matched any words, just return the first few chunks as a fallback.
        if not any(score > 0 for score, _ in ranked[:k]):
            top_chunks = self.chunks[:k]

        # Wrap raw strings into Document objects for API compatibility.
        return [LocalDocument(page_content=chunk) for chunk in top_chunks]


@lru_cache(maxsize=1)
def _get_openai_embeddings():
    """Initializes and caches the Azure OpenAI Embeddings client."""
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=get_embedding_model_name(),
        openai_api_version=get_azure_api_version(),
        azure_endpoint=get_azure_endpoint(),
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
    )


def _current_backend() -> str:
    """Determines if the system is currently configured for OpenAI or Local fallback."""
    return "openai" if has_valid_openai_key() else "local"


def _manifest_payload(source_manifest: dict | None, backend: str | None = None) -> dict:
    """Constructs the JSON-serializable manifest structure."""
    return {
        "backend": backend or _current_backend(),
        "source_manifest": source_manifest or {},
    }


def _save_manifest(source_manifest: dict | None, backend: str | None = None) -> None:
    """Saves the identifying manifest to disk to allow for future 'cache hit' checks."""
    with open(MANIFEST_FILE, "w", encoding="utf-8") as file:
        json.dump(
            _manifest_payload(source_manifest, backend),
            file,
            indent=2,
            sort_keys=True,
        )


def _load_manifest() -> dict | None:
    """Loads and returns the manifest from disk if it exists."""
    if not os.path.exists(MANIFEST_FILE):
        return None

    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def _create_local_store(chunks, source_manifest: dict | None = None):
    """Initializes a local JSON-based 'database' for offline use."""
    os.makedirs(EMBEDDING_PATH, exist_ok=True)

    with open(LOCAL_STORE_FILE, "w", encoding="utf-8") as file:
        json.dump({"chunks": chunks}, file, indent=2)

    _save_manifest(source_manifest, backend="local")
    return LocalVectorStore(chunks)


def create_vector_store(chunks, source_manifest: dict | None = None):
    """
    Main entry point for creating a searchable index from text chunks.
    It attempts to use FAISS (AI-based) but falls back to the Local Store if keys are missing.
    """
    if has_valid_openai_key():
        try:
            # FAISS is a library for efficient similarity search of dense vectors.
            from langchain_community.vectorstores import FAISS
            # Use tiktoken to estimate the cost of embedding generation before sending to the API.
            import tiktoken
            # Import the global usage tracker to record these embedding tokens.
            from utils.usage_tracker import usage_tracker

            # Calculate total tokens for the embedding request to ensure accurate billing reports.
            # We use cl100k_base which is standard for current OpenAI/Azure models.
            encoding = tiktoken.get_encoding("cl100k_base")
            total_embedding_tokens = sum(len(encoding.encode(chunk)) for chunk in chunks)
            
            # Record the usage immediately.
            usage_tracker.record_embedding_usage(total_embedding_tokens)

            embeddings = _get_openai_embeddings()
            # Convert text chunks into numerical vectors and index them.
            db = FAISS.from_texts(chunks, embeddings)
            os.makedirs(EMBEDDING_PATH, exist_ok=True)
            # Save the binary index files to disk.
            db.save_local(EMBEDDING_PATH)
            # Record that this was an OpenAI-backed index.
            _save_manifest(source_manifest, backend="openai")
            return db
        except Exception as exc:
            # If AI embeddings fail (throttling, network), log and redirect to local.
            print(
                f"OpenAI embeddings unavailable ({type(exc).__name__}). "
                "Using local retrieval fallback."
            )

    return _create_local_store(chunks, source_manifest)


def load_vector_store(source_manifest: dict | None = None):
    """
    Loads an existing index from the disk.
    It checks the manifest to decide which loader (FAISS or Local) to use.
    """
    if not vector_store_exists():
        raise FileNotFoundError("Embeddings not found. Run creation first.")

    # Prevent potential bugs by ensuring the requested PRD matches the indexed PRD.
    if source_manifest is not None and not vector_store_matches(source_manifest):
        raise FileNotFoundError(
            "Existing embeddings do not match the current PRD source set."
        )

    metadata = _load_manifest() or {}
    backend = metadata.get("backend", _current_backend())

    if backend == "openai":
        from langchain_community.vectorstores import FAISS

        embeddings = _get_openai_embeddings()
        # Load the binary vector files from the embeddings directory.
        return FAISS.load_local(
            EMBEDDING_PATH,
            embeddings,
            allow_dangerous_deserialization=True, # Required by Langchain for local loading.
        )

    # Standard path check for local fallback mode.
    if not os.path.exists(LOCAL_STORE_FILE):
        raise FileNotFoundError("Local vector store not found. Run creation first.")

    with open(LOCAL_STORE_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return LocalVectorStore(payload.get("chunks", []))


def vector_store_exists():
    """Simple check to see if any persistence files are present on the disk."""
    return (
        os.path.exists(INDEX_FILE) and os.path.exists(PICKLE_FILE)
    ) or os.path.exists(LOCAL_STORE_FILE)


def vector_store_matches(source_manifest: dict | None) -> bool:
    """
    Determines if the current source PRD content is identical to what's cached in the manifest.
    This is what allows the framework to skip 'Creating embeddings' on subsequent runs.
    """
    if not vector_store_exists():
        return False

    expected = _manifest_payload(source_manifest)
    saved_manifest = _load_manifest()
    return saved_manifest == expected
