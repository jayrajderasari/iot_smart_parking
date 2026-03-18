"""
Centralised logging configuration for Smart Parking System.

All modules obtain their logger via:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

Call setup_logging() exactly once at process start (done in main.py).
Log files are written to logs/ with a timestamped filename so every run
produces a separate, searchable file – useful on HPC batch systems.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ─── public symbols ───────────────────────────────────────────────────────────
__all__ = ["setup_logging", "get_logger"]

_LOG_DIR = Path("logs")
_CONSOLE_LEVEL = logging.INFO
_FILE_LEVEL = logging.DEBUG
_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str | Path = _LOG_DIR,
                  console_level: int = _CONSOLE_LEVEL,
                  file_level: int = _FILE_LEVEL) -> Path:
    """
    Configure root logger with a console handler and a timestamped file handler.

    Args:
        log_dir:       Directory to write log files (created if absent).
        console_level: Minimum level printed to stderr (default INFO).
        file_level:    Minimum level written to file  (default DEBUG).

    Returns:
        Path: The log file that was opened.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"smart_parking_{timestamp}.log"

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    # File handler – DEBUG and above, so every detail is preserved
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)

    # Console handler – INFO and above to avoid noise on HPC stdout
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))

    # Avoid duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)
    else:
        # Replace existing handlers (idempotent re-init)
        root.handlers.clear()
        root.addHandler(fh)
        root.addHandler(ch)

    # Reduce noise from third-party libraries
    for noisy in ("ultralytics", "torch", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("Logging initialised  →  %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.  setup_logging() must have been called first."""
    return logging.getLogger(name)
