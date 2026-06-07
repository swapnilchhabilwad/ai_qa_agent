import os  # Used for checking file existence on the disk.
from html.parser import HTMLParser  # Built-in library to parse HTML tags manually.
import requests  # Popular library for making HTTP requests (used for Confluence API).
from requests.auth import HTTPBasicAuth  # Handles authentication headers for Confluence.
from pypdf import PdfReader  # Modern library to extract text from PDF documents.
from utils.config import get_required_env  # Helper to fetch credentials from environment.


class _HTMLTextExtractor(HTMLParser):
    """
    Primitive HTML-to-Text converter. 
    It identifies structural tags (like <p> or <div>) and inserts newlines to maintain readability.
    """
    # Tags that should trigger a newline to preserve the visual document structure.
    BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []  # List to store text fragments.
        self.skip_depth = 0         # Counter to handle nested tags that should be ignored (like <script>).

    def handle_starttag(self, tag: str, attrs) -> None:
        """Called when the parser encounters a start tag (e.g., <div>)."""
        # If we hit script or style tags, stop collecting text until they close.
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return

        # If it's a structural tag, add a newline.
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Called when the parser encounters an end tag (e.g., </div>)."""
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Called when the parser encounters text content inside or between tags."""
        if self.skip_depth:
            return  # Ignore text inside <script> or <style>.

        if data.strip():
            self.parts.append(data)  # Collect meaningful text content.

    def get_text(self) -> str:
        """Joins all parts and cleans up the resulting string."""
        text = "".join(self.parts)
        # Remove extra whitespace from each line and filter out empty lines.
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


# ==============================
# 📄 LOAD PDF
# ==============================
def load_prd(file_path: str) -> str:
    """Reads a PDF file and returns its entire text content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PRD file not found: {file_path}")

    # Use PdfReader to open the file handle.
    reader = PdfReader(file_path)
    text = ""

    # Iterate through every page of the PDF.
    for page in reader.pages:
        # Extract text from the page. Fallback to empty string if extraction fails.
        text += page.extract_text() or ""

    return text


# ==============================
# 📄 LOAD TXT
# ==============================
def load_txt(file_path: str) -> str:
    """Reads a simple text file and returns its content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Open the file with UTF-8 encoding to support special characters.
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# 🌐 LOAD CONFLUENCE PAGE
# ==============================
def load_confluence_page(page_id: str) -> str:
    """
    Fetches the raw structured body of a Confluence page using its REST API.
    """
    # Fetch required settings from environment variables.
    base_url = get_required_env("CONFLUENCE_URL")
    email = get_required_env("EMAIL")
    token = get_required_env("API_TOKEN")

    # Construct the API endpoint URL for content retrieval.
    url = f"{base_url}/wiki/rest/api/content/{page_id}?expand=body.storage"

    # Execute the GET request with Basic Auth.
    response = requests.get(
        url,
        auth=HTTPBasicAuth(email, token),
        headers={"Accept": "application/json"},
    )

    # Check if the request was successful.
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Confluence page: {response.status_code}")

    # Parse the JSON response and return the HTML-like content of the page body.
    data = response.json()
    return data["body"]["storage"]["value"]


# ==============================
# 🧹 CLEAN HTML → TEXT
# ==============================
def clean_html(html: str) -> str:
    """
    Uses BeautifulSoup to strip HTML tags and extract clean, readable text.
    This is the modern, preferred way to handle web content in this framework.
    """
    from bs4 import BeautifulSoup  # Lazy import to avoid loading this large library if not needed.

    # Parse the HTML string.
    soup = BeautifulSoup(html, "html.parser")

    # Iterate through and delete unwanted tags that contain non-human-readable code.
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract all text, using newlines to separate content from different tags.
    text = soup.get_text(separator="\n")

    return text.strip()


def _legacy_clean_html_unused(html: str) -> str:
    """Older, tag-based cleaner used before BeautifulSoup was integrated."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text().strip()
