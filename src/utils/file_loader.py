import os
from utils.logger import log_info, log_warning, log_error

def load_text_files(folder_path: str) -> list[str]:
    """Return a list of raw document strings from all .txt files in folder_path."""
    docs: list[str] = []
    try:
        if not os.path.isdir(folder_path):
            log_warning(f"Folder '{folder_path}' does not exist.")
            return docs

        # deterministic order (useful for debugging)
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".txt"):
                path = os.path.join(folder_path, filename)
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())

        log_info(f"Loaded {len(docs)} .txt file(s) from '{folder_path}'")
        return docs
    except Exception as e:
        log_error(f"Error while loading files: {e}")
        return docs
