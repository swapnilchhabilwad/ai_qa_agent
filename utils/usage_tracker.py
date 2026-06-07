import json  # Used for saving and loading cumulative usage data.
import os    # Used for path manipulation and ensuring the data directory exists.
import csv   # Used for writing the transactional audit log (Manager-friendly format).
from datetime import datetime  # Used for precise timestamps and daily organization.

# Import configuration helpers to access the user-defined daily token budget and pricing rates.
from utils.config import get_daily_token_limit, get_pricing_rates

# Path to the local persistence files.
USAGE_LOG_FILE = "data/usage_log.json"
BILLING_HISTORY_FILE = "data/billing_history.csv"


class TokenUsageTracker:
    """
    Manages the recording and financial reporting of OpenAI token usage.
    It provides real-time audit logging to CSV after every single AI interaction, 
    linked to the specific source document being processed.
    """

    def __init__(self):
        """Initializes the tracker and prepares the persistence files."""
        # Ensure the data directory exists before we try to write any files to it.
        os.makedirs(os.path.dirname(USAGE_LOG_FILE), exist_ok=True)
        # Initialize session-specific counters for tokens and total cost.
        self.session_tokens = 0
        self.session_cost = 0.0
        # Tracks the current active PRD file name for entry labeling.
        self.current_source = "Unknown"
        # Tracks the active project name for scoping and billing.
        self.current_project = "default"
        # Load existing usage data from the disk.
        self._load_usage()
        # Ensure the CSV history file has the correct headers.
        self._initialize_history_file()

    def set_current_source(self, file_path: str):
        """Sets the active document context so all recorded usage is attributed to it."""
        # Store just the filename rather than the full path to keep the log clean.
        self.current_source = os.path.basename(file_path)

    def _initialize_history_file(self):
        """Creates the billing history CSV with headers if it does not already exist."""
        if not os.path.exists(BILLING_HISTORY_FILE):
            with open(BILLING_HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Timestamp: Exact date and time of the usage.
                # Source: Which PRD file was being processed.
                # Task: What the AI was doing (e.g., Analysis, Test Gen).
                # Estimated_Cost_USD: Financial impact of this specific task.
                writer.writerow([
                    "Timestamp", 
                    "Project",
                    "Source",
                    "Task", 
                    "Prompt_Tokens", 
                    "Completion_Tokens", 
                    "Total_Tokens", 
                    "Estimated_Cost_USD"
                ])

    def _load_usage(self):
        """Reads the local JSON log and pulls the data for the current day."""
        self.today = datetime.now().strftime("%Y-%m-%d")
        
        if not os.path.exists(USAGE_LOG_FILE):
            self.usage_data = {}
        else:
            try:
                with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
                    self.usage_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                self.usage_data = {}

        # If today's date exists but was created by an older version of the tracker,
        # we ensure all new keys are present to avoid KeyErrors during reporting.
        day_data = self.usage_data.setdefault(self.today, {})
        day_data.setdefault("prompt_tokens", 0)
        day_data.setdefault("completion_tokens", 0)
        day_data.setdefault("embedding_tokens", 0)
        day_data.setdefault("total_tokens", 0)
        day_data.setdefault("estimated_cost_usd", 0.0)

    def _save_usage(self):
        """Writes the current runtime usage statistics back to the physical JSON file."""
        with open(USAGE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.usage_data, f, indent=4)

    def _log_to_history_csv(self, task_label: str, prompt: int, completion: int, total: int, cost: float):
        """
        Appends a granular transaction row to the CSV audit log.
        Now includes the 'Source' column to identify the specific PRD file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(BILLING_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                self.current_project,
                self.current_source, 
                task_label, 
                prompt, 
                completion, 
                total, 
                f"${cost:.6f}"
            ])

    def _calculate_incremental_cost(self, prompt: int, completion: int, embedding: int) -> float:
        """Helper to calculate the cost in USD for a specific set of tokens."""
        rates = get_pricing_rates()
        cost = (prompt * rates["prompt"]) + \
               (completion * rates["completion"]) + \
               (embedding * rates["embedding"])
        return cost

    def record_usage(self, task_label: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """
        Adds new chat token usage and immediately logs it to the date-time history.
        """
        cost = self._calculate_incremental_cost(prompt_tokens, completion_tokens, 0)
        
        # Update session counters.
        self.session_tokens += total_tokens
        self.session_cost += cost
        
        # Update daily totals.
        day_data = self.usage_data[self.today]
        day_data["prompt_tokens"] += prompt_tokens
        day_data["completion_tokens"] += completion_tokens
        day_data["total_tokens"] += total_tokens
        day_data["estimated_cost_usd"] += cost
        
        self._save_usage()
        self._log_to_history_csv(task_label, prompt_tokens, completion_tokens, total_tokens, cost)

    def record_embedding_usage(self, token_count: int, task_label: str = "Embedding Generation"):
        """Records usage and financial impact for vector embeddings."""
        cost = self._calculate_incremental_cost(0, 0, token_count)
        
        # Update session counters.
        self.session_tokens += token_count
        self.session_cost += cost
        
        # Update daily totals.
        day_data = self.usage_data[self.today]
        day_data["embedding_tokens"] += token_count
        day_data["total_tokens"] += token_count
        day_data["estimated_cost_usd"] += cost
        
        self._save_usage()
        self._log_to_history_csv(task_label, token_count, 0, token_count, cost)

    def get_session_summary(self) -> dict:
        """Calculates a report comparing current usage against the daily budget and limits."""
        daily_limit = get_daily_token_limit()
        daily_total = self.usage_data[self.today]["total_tokens"]
        daily_cost = self.usage_data[self.today]["estimated_cost_usd"]
        remaining = max(0, daily_limit - daily_total)

        return {
            "session_tokens": self.session_tokens,
            "session_cost_usd": self.session_cost,
            "daily_tokens_used": daily_total,
            "daily_cost_usd": daily_cost,
            "daily_limit": daily_limit,
            "remaining_tokens": remaining
        }


# Singleton instance shared across the framework.
usage_tracker = TokenUsageTracker()
