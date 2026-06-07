import argparse
import hashlib  # Used for generating unique hashes of content to detect changes and manage caching.
import sys      # Provides access to system-specific parameters and functions (e.g., handling command line arguments).
import os       # Used for interacting with the operating system, like checking file paths and joining directories.

# Import specific loader functions for different PRD formats (PDF, Confluence, Text) and HTML cleaning.
from utils.loader import (
    load_prd,
    load_confluence_page,
    clean_html,
    load_txt,
)
# Import chunking logic to split large text into smaller, manageable parts for the LLM.
from utils.chunking import chunk_text
# Import utility functions for environment loading and OpenAI API key validation.
from utils.config import has_valid_openai_key, load_environment
# Import vector store management functions for indexing and searching document chunks.
from utils.vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_matches,
)
# Import core agent functions for analyzing requirements and assessing functional impact.
from agents.analyzer import analyze_requirement, analyze_impact
# Import the test case generation logic.
from agents.test_generator import generate_test_cases
# Import MemoryStore for persistent storage of analyzed features between runs.
from utils.memory import MemoryStore
# Import CSV export utility to save the generated test cases to a file.
from utils.exporter import export_to_csv
# Import the token usage tracker to report on consumption metrics at the end of the run.
from utils.usage_tracker import usage_tracker

# Configure stdout and stderr to handle UTF-8 characters properly
# This prevents crashes when printing special UI characters (like emojis) to the terminal.
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8")  # Forces standard output to use UTF-8 encoding.
    sys.stderr.reconfigure(encoding="utf-8")  # Forces standard error to use UTF-8 encoding.


def get_configured_sources(file_path: str) -> list[tuple[str, str]]:
    """
    Determines the file type (txt, pdf) based on its extension.
    Returns a list of tuples formatted as [(source_type, file_path)].
    Allows the framework to know how to parse the given input file.
    """
    sources: list[tuple[str, str]] = []  # Initialize an empty list to hold source metadata.

    # Check the file extension to decide the parser type.
    if file_path.endswith(".txt"):
        sources.append(("txt", file_path))  # Map .txt files to the text loader.
    elif file_path.endswith(".pdf"):
        sources.append(("pdf", file_path))  # Map .pdf files to the PDF loader.
    else:
        # If the format is unknown, raise an error to inform the user.
        raise ValueError(f"Unsupported file type for {file_path}")

    return sources  # Return identified sources.


def load_data(file_path: str) -> tuple[str, dict]:
    """
    Loads text content from the specified PRD file path.
    Returns:
    - combined_text: The complete formatted text extracted from the file.
    - source_manifest: A dictionary used to generate a unique hash for caching. 
    """
    sections: list[str] = []      # List to collect content from different sources if multiple are used.
    source_labels: list[str] = []  # List to track labels of sources processed.

    # Iterate through our properly formatted source tuple list (currently handles single file inputs).
    for source_type, source_value in get_configured_sources(file_path):
        if source_type == "txt":
            print(f"Loading TXT PRD: {source_value}")
            text = load_txt(source_value)  # Read raw text from .txt file.
            label = f"txt:{source_value}"  # Label for internal tracking.
        elif source_type == "pdf":
            print(f"Loading PDF PRD: {source_value}")
            text = load_prd(source_value)  # Extract text from .pdf file using PyPDF2.
            label = f"pdf:{source_value}"  # Label for internal tracking.
        else:
            # Placeholder for future multi-source expansions (e.g., Confluence).
            print(f"Loading Confluence PRD: page {source_value}")
            html = load_confluence_page(source_value)
            text = clean_html(html)
            label = f"confluence:{source_value}"

        cleaned_text = text.strip()  # Remove leading/trailing whitespace.
        if cleaned_text:
            # Format sections with source labels for better LLM context.
            sections.append(f"[SOURCE: {label}]\n{cleaned_text}")
            source_labels.append(label)

    # If no content was loaded, stop the process.
    if not sections:
        raise ValueError("Loaded PRD content is empty for the configured source set.")

    # Merge all source content into a single searchable text block.
    combined_text = "\n\n".join(sections)
    
    # Store a SHA-256 hash of the content to see if it changed since the last run.
    # This prevents rebuilding the vector store repeatedly if the file is identical.
    source_manifest = {
        "sources": source_labels,
        "text_hash": hashlib.sha256(combined_text.encode("utf-8")).hexdigest(),
    }

    print(f"Loaded {len(source_labels)} PRD source(s).")
    return combined_text, source_manifest  # Return merged text and fingerprint.


def init_vector_store(text: str, source_manifest: dict):
    """
    Initializes the local vector embeddings store.
    It checks the source manifest hash: if the file hasn't changed, it loads
    existing embeddings from disk. Otherwise, it rebuilds them.
    This optimization saves time and API costs (OpenAI Embedding calls).
    """
    # Split text into chunks so they fit within LLM context windows and allow granular retrieval.
    chunks = chunk_text(text)

    # If the text hasn't changed (hash matches stored metadata), reuse existing FAISS index.
    if vector_store_matches(source_manifest):
        print("Loading existing embeddings...")
        try:
            return load_vector_store(source_manifest)
        except Exception as exc:
            # If the index file is corrupted or missing, fallback to rebuilding.
            print(f"Existing vector store unavailable ({type(exc).__name__}). Rebuilding it now.")

    # Generate fresh embeddings using OpenAI Embeddings API or local fallback.
    print("Creating embeddings...")
    return create_vector_store(chunks, source_manifest)


def run_agent(query: str, file_path: str, project_name: str = "default"):
    """
    Main orchestration logic:
    1. Loads the file from disk.
    2. Retrieves vector chunks for semantic context.
    3. Extracts PRD features (Requirement Analysis).
    4. Evaluates impact against saved historical features (Impact Analysis).
    5. Generates exhaustive Test Cases based on the analyzed requirements.
    """
    # Link all AI usage from this point forward to the specific PRD file and project.
    usage_tracker.current_project = project_name
    usage_tracker.set_current_source(file_path)
    
    # Step A: Load the file data and its identifying manifest.
    text, source_manifest = load_data(file_path)
    # Step B: Setup the vector database for text retrieval.
    db = init_vector_store(text, source_manifest)

    # Pull the top 20 relevant chunks from the PRD to build a rich context for the AI.
    docs = db.similarity_search(query, k=20)
    context = " ".join(doc.page_content for doc in docs)

    print("\nRetrieved context ready.\n")

    # ==========================
    # Step 1: Feature Extraction
    # ==========================
    # Ask the AI to read the dense PRD text and summarize it into structural requirements.
    print("\n=== ANALYSIS ===\n")
    analysis = analyze_requirement(context)
    print("New PRD Analysis Completed.")

    # ==========================
    # Step 2: Impact Analysis
    # ==========================
    # Use MemoryStore to check if this new PRD affects previously analyzed features.
    memory_store = MemoryStore(project_name=project_name)
    
    # Get previously stored features EXCEPT the current file (prevents ego-comparison).
    historical_context = memory_store.get_historical_context(exclude_file_id=file_path)
    
    impact_analysis = ""  # Default to no impact.
    if historical_context:
        print("\n=== IMPACT ANALYSIS ===\n")
        # Ask AI: "Does this new functionality break or change existing features?"
        impact_analysis = analyze_impact(analysis, historical_context)
        if impact_analysis:
            print(f"Impact Found:\n{impact_analysis}")
        else:
            print("No older functionally impacted!")
    
    # ==========================
    # Step 3: Test Generation
    # ==========================
    # Generate test cases using the extracted features and the impact assessment.
    results = generate_test_cases(analysis, impact_analysis=impact_analysis)
    
    # ==========================
    # Step 4: Memory Commit
    # ==========================
    # Save the current analysis results into MemoryStore for future impact assessments.
    memory_store.add_or_update_feature_analysis(file_path, analysis)
    print("\n[Memory Updated] Agent safely committed PRD to System Memory.")
    
    return results  # Return the final generated test case string.


# ==============================
# 🚀 ENTRY POINT
# ==============================

if __name__ == "__main__":
    # Load environment variables (API Keys, config) from the .env file.
    load_environment()
    parser = argparse.ArgumentParser(description="Generate test cases based on a PRD document.")
    parser.add_argument(
        "--query",
        type=str,
        default="Generate comprehensive test cases exhaustively for all functional capabilities",
        help="The query or feature to generate test cases for.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/Projects/Greenprint/Onboarding/prd1.txt",
        help="The path to the PRD document (e.g., data/Projects/Greenprint/Onboarding/prd1.txt).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Greenprint",
        help="The project name to scope memory and results (e.g., Greenprint).",
    )
    args = parser.parse_args()

    query = args.query
    file_path = args.file_path
    project_name = args.project

    # Verify that the target file exists before starting the pipeline.
    if not os.path.exists(file_path):
        print(f"Error: Could not find PRD document at {file_path}")
        sys.exit(1)

    # Inform the user if the framework is running in AI mode or standard fallback mode.
    if has_valid_openai_key():
        print("Runtime mode: OpenAI-enabled")
    else:
        print("Runtime mode: local fallback (Generic responses)")

    try:
        # Launch the core execution loop.
        result = run_agent(query, file_path, project_name=project_name)
    except (RuntimeError, ValueError) as exc:
        # Catch and report any framework errors gracefully.
        print(f"\nFramework error: {exc}")
        raise SystemExit(1) from exc

    # Print the resulting test cases to the console for immediate review.
    print("\n===== 🧪 GENERATED TEST CASES =====\n")
    print(result)

    # 📤 Export Results to CSV
    # Extract the base filename (excluding extension) to name the output file.
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_filename = f"{base_name}_test_cases.csv"
    
    # Determine Module from file_path if possible (e.g. data/Projects/Greenprint/Onboarding/prd.txt)
    path_parts = os.path.normpath(file_path).split(os.sep)
    module_name = ""
    if "Projects" in path_parts:
        idx = path_parts.index("Projects")
        if len(path_parts) > idx + 2:
            module_name = path_parts[idx + 2]
            # If the next part is the file itself, then there's no module directory
            if module_name == os.path.basename(file_path):
                module_name = ""

    # Define the output directory path for the results.
    results_dir = os.path.join("data", "Results")
    if project_name != "default":
        results_dir = os.path.join(results_dir, project_name)
    
    if module_name:
        results_dir = os.path.join(results_dir, module_name)
        
    # Ensure the directory exists; if not, create it.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    csv_path = os.path.join(results_dir, csv_filename)
    
    # Save the generated tests to CSV for sharing or importing into test management tools.
    export_to_csv(result, csv_path)

    # 📊 Final AI Usage & Billing Report
    # Retrieve the summary of tokens and financial costs used in this session and throughout the day.
    summary = usage_tracker.get_session_summary()
    print("\n" + "="*45)
    print("📈 AZURE OPENAI USAGE & BILLING REPORT")
    print("="*45)
    print(f"Session Tokens:     {summary['session_tokens']:,}")
    print(f"Session Cost:       ${summary['session_cost_usd']:.4f}")
    print("-" * 45)
    print(f"Daily Total Tokens: {summary['daily_tokens_used']:,}")
    print(f"Daily Total Cost:   ${summary['daily_cost_usd']:.4f}")
    print(f"Daily Token Limit:  {summary['daily_limit']:,}")
    print(f"Remaining (Budget): {summary['remaining_tokens']:,}")
    print("="*45)
    
    # Notify the user where they can find the persistent usage and billing log.
    print(f"\nPersistent billing log updated at: data/usage_log.json")