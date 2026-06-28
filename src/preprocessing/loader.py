"""
Data loading, extraction, cleaning, and splitting module.

Supports CSV, TXT, ZIP, and GZ files. Auto-detects the NSL-KDD format
(43 columns, no header) and applies correct column names. Uses chunked
reading to handle datasets larger than available RAM.
"""

import pandas as pd
import numpy as np
import zipfile
import gzip
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from configs.config import DATA_RAW_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, VAL_SIZE, CHUNKSIZE


# ---------------------------------------------------------------------------
# NSL-KDD column names (in order)
# ---------------------------------------------------------------------------
# The NSL-KDD dataset has 42 features + 1 label + 1 difficulty score.
# We drop the difficulty column after loading.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]


def _find_parquet() -> list[Path]:
    """Return a list of all Parquet files found in the raw data directory."""
    return list(DATA_RAW_DIR.glob("*.parquet"))


def _find_archives() -> list[Path]:
    """Return a list of all ZIP and GZ files found in the raw data directory."""
    return list(DATA_RAW_DIR.glob("*.zip")) + list(DATA_RAW_DIR.glob("*.gz"))


def _extract_if_needed(filepath: Path) -> Path:
    """
    Extract a ZIP or GZ archive and return the path(s) to the CSVs inside.

    For ZIP files, extracts into a sub-directory named after the archive
    (e.g., NSL-KDD.zip -> data/raw/NSL-KDD/). For GZ files, decompresses
    in-place by stripping the .gz extension.

    When multiple CSV files exist under the extracted directory, returns
    the directory itself so that load_data can concatenate all of them
    (datasets like CIC_IoMT_2024 split classes across train/test files).

    The extraction is skipped if the output directory/file already exists,
    making repeated calls cheap.
    """
    # Handle ZIP archives
    if filepath.suffix == ".zip":
        extract_dir = DATA_RAW_DIR / filepath.stem
        if not extract_dir.exists():
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(extract_dir)
            print(f"[loader] Extracted {filepath.name} -> {extract_dir}")
        # If multiple CSVs exist, return the directory so load_data
        # can concatenate them all (provides class diversity)
        csv_files = list(extract_dir.rglob("*.csv")) + list(extract_dir.rglob("*.txt"))
        if csv_files:
            if len(csv_files) > 1:
                print(f"[loader] Found {len(csv_files)} data files, will concatenate all")
                return extract_dir
            chosen = csv_files[0]
            print(f"[loader] Selected file: {chosen.name}")
            return chosen
        raise FileNotFoundError(f"No CSV/TXT found inside {filepath.name}")

    # Handle GZIP archives
    if filepath.suffix == ".gz":
        csv_path = filepath.with_suffix("")
        if not csv_path.exists():
            with gzip.open(filepath, "rb") as f_in:
                with open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"[loader] Decompressed {filepath.name} -> {csv_path.name}")
        return csv_path

    # Not an archive — return the path as-is
    return filepath


def _detect_header_and_columns(filepath: Path, nrows: int = 5) -> tuple:
    """
    Detect whether a dataset file has a header row and what column names to use.

    Reads the first few rows and applies heuristics:
    - TXT files with 43 columns are treated as NSL-KDD (no header).
    - Files whose first row contains only numeric values likely have no header.
    - Otherwise, assume the first row is a header.

    Returns (header, names) where:
    - header = 0  -> first row is a header (use it as column names)
    - header = None -> no header (use provided `names` or auto-generate integers)
    - names = list of column names, or None to let pandas infer them
    """
    # Read a small sample to inspect the file structure
    sample = pd.read_csv(filepath, nrows=nrows)
    n_columns = sample.shape[1]

    # Heuristic 1: .txt file with 43 columns -> NSL-KDD, no header
    is_txt = filepath.suffix.lower() == ".txt"
    if is_txt and n_columns == 43:
        print("[loader] Detected NSL-KDD 43-column format")
        return None, NSL_KDD_COLUMNS[:43]

    # Heuristic 2: column names are integers -> pandas treated first row as data
    first_col = str(sample.columns[0]).lower()
    is_numeric_headers = sample.columns.dtype in ("int64", "int32", "float64")
    if is_numeric_headers:
        if n_columns == 43:
            # 43 numeric columns is almost certainly NSL-KDD read without header
            print("[loader] Detected NSL-KDD 43-column format (numeric headers)")
            return None, NSL_KDD_COLUMNS[:43]
        print(f"[loader] No header detected ({n_columns} columns)")
        return None, list(range(n_columns))

    # Heuristic 3: all column names are strings, and one of them looks like
    # a feature name -> there IS a header, let pandas use it
    all_cols_are_strings = all(isinstance(c, str) for c in sample.columns)
    has_label_header = any(kw in first_col for kw in ["duration", "protocol", "label", "id", "feature"])
    if all_cols_are_strings and (has_label_header or n_columns != 43):
        return 0, None

    # Fallback: no header detected
    print(f"[loader] No header detected ({n_columns} columns)")
    return None, list(range(n_columns))


def find_dataset() -> Path:
    """
    Locate a dataset file in data/raw/, extracting archives if necessary.

    Priority: Parquet (fastest, most memory-efficient), ZIP/GZ archives,
    then CSV, then TXT. Returns the path to the first usable file found.
    """
    # Prefer Parquet — 10-15x smaller, much faster to load
    parquet_files = _find_parquet()
    if parquet_files:
        train_files = [f for f in parquet_files if "train" in f.stem.lower()]
        if train_files:
            return train_files[0]
        return parquet_files[0]

    archives = _find_archives()
    if archives:
        return _extract_if_needed(archives[0])

    csv_files = list(DATA_RAW_DIR.glob("*.csv")) + list(DATA_RAW_DIR.glob("*.txt"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV/TXT/ZIP/GZ/Parquet files found in {DATA_RAW_DIR}. "
            "Place your dataset in data/raw/ and re-run."
        )
    return csv_files[0]


def load_data(filepath: Path = None, chunksize: int = None, max_rows: int = None) -> pd.DataFrame:
    """
    Load a dataset from disk using chunked reading to limit memory usage.

    Steps:
    1. Locate the dataset file (auto-extract if needed).
    2. Detect header/column format.
    3. Read data — uses Parquet if available (fast, memory-efficient),
       otherwise falls back to chunked CSV reading.
    4. When max_rows is set, load a large enough pool to find diverse
       classes (the file may be ordered by label), then randomly sample.
    5. Drop the 'difficulty' column if present (NSL-KDD).

    Args:
        filepath: Path to dataset. If None, auto-detect.
        chunksize: Rows per chunk for CSV fallback. If None, uses config value.
        max_rows: Target number of rows. Loads a larger pool internally
                  for class diversity, then randomly samples max_rows.

    Returns:
        DataFrame containing the loaded data.
    """
    if filepath is None:
        filepath = find_dataset()
    else:
        filepath = _extract_if_needed(filepath)

    chunksize = chunksize or CHUNKSIZE

    # If filepath is a directory, pick the training CSV (it has multiple
    # classes from a single source, making classification realistic).
    if filepath.is_dir():
        parquet_files = list(filepath.rglob("*.parquet"))
        csv_files = list(filepath.rglob("*.csv")) + list(filepath.rglob("*.txt"))
        # Prefer parquet inside the directory too, then training CSV
        if parquet_files:
            train_files = [f for f in parquet_files if "train" in f.stem.lower()]
            filepath = train_files[0] if train_files else parquet_files[0]
        else:
            train_files = [f for f in csv_files if "train" in f.stem.lower()]
            filepath = train_files[0] if train_files else csv_files[0]
        print(f"[loader] Using training file: {filepath.name}")

    print(f"[loader] Loading dataset: {filepath}  ({filepath.stat().st_size / 1e6:.1f} MB)")

    # ---- Parquet path (fast, memory-efficient) ----
    if filepath.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows
        # Log file identity for cross-environment debugging
        import hashlib as _hl
        _buf = open(filepath, "rb").read(1 << 20)  # first 1 MB
        _h = _hl.md5(_buf).hexdigest()
        print(f"[loader]  Parquet rows={total_rows:,}  row_groups={pf.metadata.num_row_groups}  md5(1MB)={_h}")
        # Read a limited pool (10x max_rows) instead of the full file to save
        # memory on constrained hosts (Railway has only 512 MB RAM).
        if max_rows is not None and total_rows > max_rows:
            n_row_groups = pf.metadata.num_row_groups
            rows_per_group = max(1, total_rows // n_row_groups)
            target_pool = min(total_rows, max_rows * 10)
            groups_needed = max(1, target_pool // rows_per_group)
            if groups_needed < n_row_groups:
                import random as _random
                _random.seed(RANDOM_STATE)
                groups = sorted(_random.sample(range(n_row_groups), groups_needed))
                table = pf.read_row_groups(groups)
                print(f"[loader]  Read {groups_needed}/{n_row_groups} row groups ({len(table)} rows) from Parquet")
            else:
                table = pf.read()
                print(f"[loader]  Read all {n_row_groups} row groups ({len(table)} rows) from Parquet")
            df = table.to_pandas()
        else:
            df = pd.read_parquet(filepath, engine="pyarrow")
            print(f"[loader]  Loaded {len(df)} rows from Parquet")
        # Shuffle and sample to max_rows if needed
        if max_rows is not None:
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
            else:
                df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[loader] Final shape: {df.shape}")
        return df

    # ---- CSV / TXT path (chunked reading) ----
    # Detect whether the file has a header row and what columns to use
    header, names = _detect_header_and_columns(filepath)

    # Load enough rows to capture class diversity. Since CIC_IoMT_2024
    # has 51 classes spread across 7.16M rows (ordered by class), we
    # remove the old 1M hard cap and scale load limit to the dataset.
    # For datasets ordered by class, we load a pool large enough to
    # reach all class types, then randomly sample down.
    load_limit = max(max_rows * 10, chunksize) if max_rows else None
    if load_limit:
        print(f"[loader]  Load limit: {load_limit} rows")

    # Read the file in chunks and concatenate
    chunks = []
    rows_loaded = 0
    for chunk in pd.read_csv(filepath, header=header, names=names, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
        rows_loaded += chunk.shape[0]
        print(f"[loader]  Loaded chunk: {chunk.shape[0]} rows (total: {rows_loaded})")
        if load_limit is not None and rows_loaded >= load_limit:
            print(f"[loader]  Reached load limit, stopping early")
            break
    df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]

    # When loading from multiple files (or a single file with max_rows),
    # shuffle so the row order is randomised (files are ordered by label)
    if max_rows is not None:
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        else:
            df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # NSL-KDD has a 'difficulty' column that we don't need for classification
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    print(f"[loader] Final shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning:
    - Remove duplicate rows
    - Drop columns that are entirely NaN
    - Forward-fill remaining missing values, then fill anything left with 0
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"[loader] Dropped {before - len(df)} duplicates")
    # Drop columns where every value is missing
    df = df.dropna(axis=1, how="all")
    # Only fill missing values in numeric columns that actually have NaN.
    # Upcast float16 → float32 first to avoid pandas C-level issues with
    # half-precision floats (parquet often uses float16).
    num_cols = df.select_dtypes(include=["number", "bool"]).columns
    for c in num_cols:
        if df[c].dtype.name == "float16":
            df[c] = df[c].astype("float32")
    cat_cols = df.select_dtypes(include=["category"]).columns
    if len(cat_cols):
        for c in cat_cols:
            df[c] = df[c].cat.add_categories("").fillna("")
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(num_cols):
        cols_with_nan = [c for c in num_cols if df[c].isna().any()]
        if cols_with_nan:
            df[cols_with_nan] = df[cols_with_nan].ffill().fillna(0)
    # For string columns, fill missing with empty string
    if len(str_cols):
        df[str_cols] = df[str_cols].fillna("")
    return df


def split_data(df: pd.DataFrame):
    """
    Split a DataFrame into train / validation / test sets.

    Steps:
    1. Identify the target column (searches for 'label', 'class', 'target',
       or falls back to the last column).
    2. Encode string labels into integer codes 0..N-1.
    3. One-hot encode categorical feature columns.
    4. Split: 72% train, 8% validation, 20% test.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # ---- Step 1: Locate the target column ----
    if TARGET_COLUMN not in df.columns:
        # Search for common target column names
        possible = [c for c in df.columns if isinstance(c, str) and (
            "label" in c.lower() or "class" in c.lower() or "target" in c.lower()
        )]
        if possible:
            target = possible[0]
            print(f"[loader] Target column '{TARGET_COLUMN}' not found, using '{target}' instead")
        else:
            # Last resort: use the last column
            target = df.columns[-1]
            print(f"[loader] Target column '{TARGET_COLUMN}' not found, using last column '{target}'")
    else:
        target = TARGET_COLUMN

    # Separate features (X) from target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # ---- Step 1b: Drop features highly correlated with label ----
    # These make classification trivially perfect and hide the effect of
    # adversarial attacks. Drop any numeric feature with |corr| > 0.90.
    if y.dtype.name in ("object", "str", "string", "category"):
        le = LabelEncoder()
        y_int = le.fit_transform(y)
    else:
        y_int = y
    high_corr = []
    for c in X.columns:
        if X[c].dtype.kind in ("i", "f"):
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(X[c].fillna(0).values.astype(float), y_int)[0, 1]
            if not np.isnan(corr) and abs(corr) > 0.90:
                high_corr.append(c)
    if high_corr:
        X = X.drop(columns=high_corr)
        print(f"[loader] Dropped {len(high_corr)} high-corr features: {high_corr}")

    # ---- Step 2: Encode string labels to integers ----
    # In pandas >= 2.0 string columns may have dtype 'str' (not 'object'),
    # so we check the dtype *name* to catch both.
    label_encoder = None
    is_string_like = y.dtype.name in ("object", "str", "string", "category")
    if is_string_like:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target, dtype="int64")
        print(f"[loader] Encoded labels: {len(label_encoder.classes_)} classes")
        print(f"[loader] Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # ---- Step 3: One-hot encode categorical features ----
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f"[loader] One-hot encoding {len(cat_cols)} categorical columns: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ---- Step 4: Train / validation / test split ----
    # First split: separate test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Second split: carve validation set from training (10% of training = 8% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"[loader] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
