from utils.logger import log_header, log_info, log_success
from utils.file_loader import load_text_files

if __name__ == "__main__":
    log_header("Phase 1 — File Loading Sanity Check")

    folder = "datasets/test"
    docs = load_text_files(folder)

    if not docs:
        log_info("No documents found. Add .txt files under datasets/small/ and re-run.")
    else:
        log_success("Documents loaded successfully 🎉")
        # Show a tiny preview so we know the content came through
        for i, d in enumerate(docs, start=1):
            preview = d.strip().splitlines()[0][:80]
            log_info(f"Doc {i}: \"{preview}\"")
