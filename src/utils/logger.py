import os
import datetime

# ====================================================
# Color codes for better console logging
# ====================================================
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# ====================================================
# Log directory setup
# ====================================================
LOG_DIR = "results/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "runtime.log")


def _write_to_file(level: str, message: str):
    """Internal helper to append logs to file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")


# ====================================================
# Logging functions
# ====================================================

def log_info(message: str, color: str = Colors.CYAN):
    """Log info message with color + write to file"""
    print(f"{color}ℹ️  {message}{Colors.END}")
    _write_to_file("INFO", message)


def log_success(message: str):
    """Log success message in green + write to file"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    _write_to_file("SUCCESS", message)


def log_error(message: str):
    """Log error message in red + write to file"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")
    _write_to_file("ERROR", message)


def log_warning(message: str):
    """Log warning message in yellow + write to file"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
    _write_to_file("WARNING", message)


def log_header(message: str):
    """Log section header with emphasis + write to file"""
    border = "=" * 60
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{border}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}🚀 {message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{border}{Colors.END}\n")
    _write_to_file("HEADER", message)
