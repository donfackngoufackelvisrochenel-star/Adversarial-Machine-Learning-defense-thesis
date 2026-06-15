"""
Create a small sample of any dataset in data/raw/ for quick demo runs.

Usage:
    python scripts/sample_dataset.py CICIoMT2024.csv --rows 1000

This reads the original file, takes the first N rows, and writes
data/raw/sample.csv which the pipeline/dashboard will use by default.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import DATA_RAW_DIR


def main():
    parser = argparse.ArgumentParser(description="Sample a dataset for quick demos")
    parser.add_argument("filename", help="Dataset file in data/raw/ (e.g. CICIoMT2024.csv)")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to keep")
    parser.add_argument("--output", default=None, help="Output filename (default: sample.csv)")
    args = parser.parse_args()

    src = DATA_RAW_DIR / args.filename
    if not src.exists():
        files = list(DATA_RAW_DIR.glob("*"))
        print(f"File not found: {src}")
        print(f"Available files: {[f.name for f in files]}")
        return

    dst = DATA_RAW_DIR / (args.output or f"sample_{args.rows}.csv")
    df = pd.read_csv(src, nrows=args.rows, low_memory=False)
    df.to_csv(dst, index=False)
    print(f"Sampled {args.rows} rows -> {dst} ({dst.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
