"""
Central configuration for the Adversarial Machine Learning Defense project.

All tunable parameters — paths, model hyperparameters, attack settings,
defense settings, and data-loading controls — are defined here so they
can be changed in a single place without touching any other file.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env file if present (for credentials)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Project directory structure
# ---------------------------------------------------------------------------

# BASE_DIR is the project root (parent of configs/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Where raw dataset files (CSV, ZIP, GZ) should be placed by the user
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Where processed / cleaned data is saved after preprocessing
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Where trained model pickles are stored
MODELS_DIR = BASE_DIR / "models"

# Where pipeline logs are written
LOGS_DIR = BASE_DIR / "logs"

# Where evaluation plots and CSV reports are saved
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all data directories exist (created automatically if missing)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

# Default name of the target (label) column in the dataset
TARGET_COLUMN = "label"

# Fraction of data held out for testing
TEST_SIZE = 0.2

# Random seed for reproducible train/test splits and model training
RANDOM_STATE = 42

# Fraction of the *training* set held out for validation (applied after the
# test split, so the actual split ratios are train=0.72, val=0.08, test=0.20)
VAL_SIZE = 0.1

# ---------------------------------------------------------------------------
# Row limit for demos (set to a small number to avoid OOM on large
# datasets; None = load everything)
# ---------------------------------------------------------------------------
# With 51 classes in CIC_IoMT_2024, we need enough rows per class.
# 300K rows → ~5.9K avg per class, fitting within 4 GB available RAM.
# Set to 0 to load all rows.
MAX_ROWS = 300000

# ---------------------------------------------------------------------------
# Memory management for large datasets
# ---------------------------------------------------------------------------

# Number of rows to read per chunk when loading CSVs.
# Lower values reduce peak RAM usage at the cost of slightly slower loading.
# For NSL-KDD (~126K rows) 5000 is safe; for datasets with millions of rows
# you may need to go even lower (e.g., 1000).
CHUNKSIZE = 50000

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    # Random Forest — ensemble of decision trees with bagging
    "random_forest": {
        "n_estimators": 10,         # Reduced for fast demo on Railway (default 100)
        "max_depth": 6,              # Shallow → fast training
        "min_samples_split": 10,     # Minimum samples required to split a node
        "random_state": RANDOM_STATE,
        "n_jobs": -1,                # Use all CPU cores
    },
    # XGBoost — gradient-boosted decision trees
    "xgboost": {
        "n_estimators": 15,          # Reduced for fast demo on Railway (default 100)
        "max_depth": 4,              # Shallow trees → faster training
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss",   # Multi-class log loss (works for 2+ classes)
    },
    # LightGBM — efficient gradient boosting with leaf-wise tree growth
    "lightgbm": {
        "n_estimators": 15,          # Reduced for fast demo on Railway (default 100)
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
        "verbose": -1,               # Suppress LightGBM's own training messages
    },
}

# ---------------------------------------------------------------------------
# Evasion attack parameters
# ---------------------------------------------------------------------------

ATTACK_PARAMS = {
    # FGSM — single-step gradient-based attack
    "fgsm": {
        "epsilon": 0.1,  # Perturbation magnitude (fraction of feature std-dev)
    },
    # PGD — iterative multi-step attack with projection
    "pgd": {
        "epsilon": 0.1,   # Total perturbation budget
        "alpha": 0.01,    # Step size per iteration
        "num_iter": 10,   # Number of PGD iterations
    },
}

# ---------------------------------------------------------------------------
# Defense parameters
# ---------------------------------------------------------------------------

DEFENSE_PARAMS = {
    # Adversarial training — augment training set with adversarial examples
    "adversarial_training": {
        "epsilon": 0.1,   # Attack strength used during augmentation
        "epochs": 1,      # Reduced for fast demo (default 3)
    },
    # Feature selection — keep only the K most robust features
    "feature_selection": {
        "top_k": 20,      # Number of features to retain (least vulnerable)
    },
    # Ensemble — voting strategy across multiple model types
    "ensemble": {
        "voting": "soft",  # "soft" = average probabilities; "hard" = majority vote
    },
    # Feature squeezing — reduce input precision to block subtle perturbations
    "feature_squeezing": {
        "bit_depth": 4,   # Number of quantization levels (2^bits)
    },
}

# ---------------------------------------------------------------------------
# Authentication / Security
# ---------------------------------------------------------------------------

# Credentials for API Basic Auth and dashboard login
# Change these in production — never commit real credentials.
# Values are read from environment variables (can be set via .env file).
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "ciciomt2024")

# API key for programmatic access (passed as X-API-Key header)
API_KEY = os.environ.get("API_KEY", "aml-defense-key-2024")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGING_LEVEL = "INFO"
