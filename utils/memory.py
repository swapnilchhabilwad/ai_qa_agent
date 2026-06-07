import json  # Used for serializing and deserializing the feature memory mapping to a text file.
import os    # Used for checking file paths and creating directories on the local machine.

# Define the absolute local path where the historically generated features are persisted.
# This file serves as the "long-term memory" of everything the agent has analyzed.
MEMORY_FILE = "data/system_memory.json"

class MemoryStore:
    """
    Manages the long-term system memory for the QA Agent framework.
    It associates requirement analyses (features, rules, etc.) with specific filenames
    to allow for cross-document impact assessment.
    """
    
    def __init__(self, project_name: str = "default"):
        """Initializes the store and ensures the persistence layer is ready."""
        # Determine the memory file path based on the project name.
        # "default" maps to the original system_memory.json for backward compatibility.
        if project_name == "default":
            self.memory_file = "data/system_memory.json"
        else:
            self.memory_file = f"data/{project_name}_memory.json"
            
        # Create the data/ folder proactively if it does not already exist.
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        # Load existing data from the disk into the runtime memory dictionary.
        self._load()

    def _load(self):
        """
        Internal function that reads the JSON file from the disk. 
        It handles missing files and provides migration logic for older versions.
        """
        # If no file exists, initialize an empty schema and save it to create the file.
        if not os.path.exists(self.memory_file):
            self.memory = {"historical_features": {}}
            self._save()
        else:
            try:
                # Open and parse the JSON file with UTF-8 support.
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                    
                # [Backward Compatibility / Migration] 
                # Older versions of the framework stored features as a List[].
                # New versions use a Dict[filename, text] for better overwrite tracking.
                if isinstance(self.memory.get("historical_features"), list):
                    migrated = {}
                    for i, feat in enumerate(self.memory["historical_features"]):
                        migrated[f"legacy_prd_{i}.txt"] = feat
                        
                    # Replace the old list with the newly migrated dictionary.
                    self.memory["historical_features"] = migrated
                    self._save()
            except Exception:
                # If the file is corrupted or unreadable, reset to a safe empty state.
                self.memory = {"historical_features": {}}

    def _save(self):
        """
        Serializes the current runtime dictionary and flushes it to the physical disk.
        """
        with open(self.memory_file, "w", encoding="utf-8") as f:
            # indent=4 makes the JSON file human-readable if opened in an editor.
            json.dump(self.memory, f, indent=4)

    def add_or_update_feature_analysis(self, file_id: str, analysis_text: str):
        """
        Registers a new requirement analysis into the memory bank.
        
        Using the filename (file_id) as the key ensures that if a user updates an existing
        PRD and reruns it, the old analysis for that file is updated rather than duplicated.
        """
        if not analysis_text:
            return
            
        # Store or overwrite the analysis for the given file ID.
        self.memory["historical_features"][file_id] = analysis_text
        # Immediately persist the change to disk.
        self._save()

    def get_historical_context(self, exclude_file_id: str = None) -> str:
        """
        Returns all system memory as a single formatted string for the AI's context window.
        
        Args:
            exclude_file_id: The filename currently being processed. This prevents the agent
                             from comparing a new version of a file against its own old version.
        """
        features = self.memory.get("historical_features", {})
        
        # Pull all features except those belonging to the current file being analyzed.
        filtered_features = {
            f_id: feat for f_id, feat in features.items() if f_id != exclude_file_id
        }

        # If no other files have been analyzed before, return an empty string.
        if not filtered_features:
            return ""
        
        # Build a structured string block containing all previous functionality details.
        context = "Previously Implemented System Features Context (Other PRDs):\n"
        for f_id, feat in filtered_features.items():
            context += f"\n--- Source: {f_id} ---\n{feat}\n"
            
        return context
