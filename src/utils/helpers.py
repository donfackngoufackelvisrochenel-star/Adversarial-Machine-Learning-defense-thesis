"""
Utility functions for logging, saving results, and checking data
availability.
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from configs.config import LOGS_DIR, LOGGING_LEVEL


def setup_logger(name: str = "aml") -> logging.Logger:
    """
    Create a logger that writes to both a file (logs/<name>.log) and stdout.

    Args:
        name: Logger name (also used as the log filename).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOGGING_LEVEL.upper(), logging.INFO))

    # File handler
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_results(results: dict, name: str):
    """
    Serialise a results dictionary to reports/<name>.json.

    Args:
        results: Dict of metric_name → value.
        name: Base filename (without extension).
    """
    path = Path("reports") / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[utils] Results saved to {path}")


def load_results(name: str) -> dict:
    """
    Load a previously saved results file from reports/<name>.json.

    Args:
        name: Base filename (without extension).

    Returns:
        Dict of metric_name → value.
    """
    path = Path("reports") / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def check_dataset_exists() -> bool:
    """
    Return True if at least one dataset file exists in data/raw/.

    Supports CSV, TXT, ZIP, and GZ files.
    """
    from configs.config import DATA_RAW_DIR
    patterns = ("*.csv", "*.txt", "*.zip", "*.gz")
    for p in patterns:
        if list(DATA_RAW_DIR.glob(p)):
            return True
    return False
