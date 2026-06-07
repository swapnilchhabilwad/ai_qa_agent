# Import the RecursiveCharacterTextSplitter from Langchain.
# This library is the industry standard for splitting large documents into smaller pieces while trying to keep related sentences together.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str):
    """
    Divide a long string of text (like a full PRD) into smaller, overlapping segments.
    This is necessary because LLMs have a 'context window' (a limit on how much text they can process at once).
    """

    # Initialize the splitter with specific parameters:
    # - chunk_size: Each piece will be roughly 1000 characters long.
    # - chunk_overlap: Each piece will share 200 characters with the previous and next piece.
    #   This ensures that context (like a sentence being cut in half) is preserved across chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    # Perform the split and return a list of text strings.
    return splitter.split_text(text)
