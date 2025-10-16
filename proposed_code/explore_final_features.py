"""
Exploration script for E:\\RUL\\3\\final_features.csv

What this script does (safe on medium/large CSVs):
1) Loads the CSV (with sensible defaults; tries pyarrow fast-path if available).
2) Prints: shape, dtypes, memory usage, column name checks, missingness, basic stats.
3) Detects potential ID / target / time columns.
4) Shows per-column quick summaries (top categories for object-like, range for numeric).
5) Checks duplicates, constant/near-constant cols, zero-variance, and highly correlated pairs.
6) If file is too big for memory, fallback to a chunked overview (row count, missingness %, dtype guess).

Usage:
    python explore_final_features.py
(Optional) To point to a different file:
    python explore_final_features.py --path "E:\\RUL\\3\\final_features.csv"
"""

from __future__ import annotations
import os
import sys
import argparse
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# ------------ Helpers ------------

def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

LIKELY_ID_TOKENS = {"id", "idx", "uid", "guid", "index"}
LIKELY_TARGET_TOKENS = {"rul", "label", "target", "y", "failure", "status"}
LIKELY_TIME_TOKENS = {"time", "timestamp", "ts", "date", "datetime", "t"}
LIKELY_GROUP_TOKENS = {"test", "run", "unit", "specimen", "beam", "slab", "fold"}

# ------------ Core exploration functions ------------

def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load CSV with fast engine if possible, falling back gracefully."""
    read_kwargs = dict(
        low_memory=False,
        nrows=nrows,
        engine="pyarrow" if "pyarrow" in pd.__dict__.get("options", {}).__class__.__name__.lower() or True else None,
    )
    # Prefer pyarrow engine when available (pandas>=2 often supports it via engine="pyarrow")
    try:
        return pd.read_csv(path, **{k: v for k, v in read_kwargs.items() if v is not None})
    except Exception:
        # Try the default engine
        try:
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")


def basic_info(df: pd.DataFrame) -> None:
    print("\n=== BASIC INFO ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    mem_bytes = df.memory_usage(deep=True).sum()
    print(f"Approx. in-memory size: {sizeof_fmt(mem_bytes)}")

    print("\nDtypes:")
    print(df.dtypes.sort_index())

    print("\nNon-null counts (first 50 cols if many):")
    nn = df.notna().sum().sort_values(ascending=True)
    if len(nn) > 50:
        print(nn.head(50))
    else:
        print(nn)


def guess_special_columns(columns: List[str]) -> Dict[str, List[str]]:
    ids, targets, times, groups = [], [], [], []
    for c in columns:
        lc = c.lower()
        if any(tok in lc for tok in LIKELY_ID_TOKENS):
            ids.append(c)
        if any(tok in lc for tok in LIKELY_TARGET_TOKENS):
            targets.append(c)
        if any(tok in lc for tok in LIKELY_TIME_TOKENS):
            times.append(c)
        if any(tok in lc for tok in LIKELY_GROUP_TOKENS):
            groups.append(c)
    return {"id_like": ids, "target_like": targets, "time_like": times, "group_like": groups}


def parse_time_columns(df: pd.DataFrame, time_cols: List[str]) -> List[str]:
    parsed = []
    for c in time_cols:
        try:
            parsed_col = pd.to_datetime(df[c], errors="raise", utc=False, infer_datetime_format=True)
            df[c] = parsed_col
            parsed.append(c)
        except Exception:
            pass
    return parsed


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().rename("missing_frac").to_frame()
    miss["missing_%"] = (miss["missing_frac"] * 100).round(2)
    miss["non_null_count"] = df.notna().sum()
    return miss.sort_values("missing_frac", ascending=False)


def constant_columns(df: pd.DataFrame, near_tol: float = 1e-12) -> Tuple[List[str], List[str]]:
    zero_var = []
    near_const = []
    for c in df.columns:
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            zero_var.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            rng = s.max() - s.min()
            if pd.notna(rng) and rng <= near_tol:
                near_const.append(c)
    return zero_var, near_const


def quick_column_summaries(df: pd.DataFrame, max_cat: int = 10, max_print: int = 50) -> None:
    print("\n=== QUICK COLUMN SUMMARIES ===")
    cols = list(df.columns)
    if len(cols) > max_print:
        print(f"Showing first {max_print} of {len(cols)} columns...")
        cols = cols[:max_print]
    for c in cols:
        s = df[c]
        print(f"\n-- {c} --")
        print(f"dtype={s.dtype}, non-null={s.notna().sum():,}, unique={s.nunique(dropna=True):,}")
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            print(desc)
        else:
            vc = s.value_counts(dropna=True).head(max_cat)
            print("Top categories:")
            print(vc)


def duplicates_check(df: pd.DataFrame) -> None:
    print("\n=== DUPLICATES ===")
    dup_all = df.duplicated().sum()
    print(f"Exact duplicate rows: {dup_all:,}")


def correlation_peek(df: pd.DataFrame, top_k: int = 20, thresh: float = 0.95) -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        print("\n=== CORRELATIONS ===\nNot enough numeric columns for correlation.")
        return
    print("\n=== CORRELATIONS (top pairs) ===")
    corr = num_df.corr(numeric_only=True).abs()
    # Take upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = (
        corr.where(mask)
            .stack()
            .sort_values(ascending=False)
    )
    if len(pairs) == 0:
        print("No correlation pairs found.")
        return
    print(pairs.head(top_k))
    high = pairs[pairs >= thresh]
    if len(high) > 0:
        print(f"\nHighly correlated (>= {thresh}):")
        print(high)


def per_group_overview(df: pd.DataFrame, group_cols: List[str]) -> None:
    if not group_cols:
        return
    print("\n=== PER-GROUP OVERVIEW ===")
    for g in group_cols[:2]:  # limit to first two plausible grouping columns
        gb = df.groupby(g, dropna=False)
        print(f"\nBy '{g}':")
        print("count per group (top 20):")
        print(gb.size().sort_values(ascending=False).head(20))
        # If time column exists, show min/max per group
        time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if time_cols:
            t = time_cols[0]
            agg = gb[t].agg(['min', 'max']).head(10)
            print(f"time range per '{g}' (first 10 groups):")
            print(agg)


def class_balance(df: pd.DataFrame, target_cols: List[str]) -> None:
    for y in target_cols:
        s = df[y]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            print(f"\nTarget candidate '{y}' looks continuous. Basic stats:")
            print(s.describe())
        else:
            print(f"\nClass balance for '{y}':")
            print(s.value_counts(dropna=False).head(50))


def chunked_overview(path: str, chunksize: int = 1_000_000) -> None:
    print("\n=== CHUNKED OVERVIEW (memory-safe) ===")
    total_rows = 0
    null_counts: Optional[pd.Series] = None
    sample_chunk: Optional[pd.DataFrame] = None

    for i, chunk in enumerate(pd.read_csv(path, low_memory=False, chunksize=chunksize)):
        if i == 0:
            sample_chunk = chunk.head(5)
            null_counts = chunk.isna().sum()
        else:
            null_counts = null_counts.add(chunk.isna().sum(), fill_value=0)
        total_rows += len(chunk)
        if i >= 9:
            # don't iterate forever on massive files; 10M rows sample is plenty to gauge shape
            break

    print(f"Rows scanned: {total_rows:,}")
    if sample_chunk is not None:
        print("\nSample rows (first chunk):")
        print(sample_chunk)
    if null_counts is not None:
        miss_pct = (null_counts / total_rows * 100).round(2)
        miss_report = pd.DataFrame({"nulls": null_counts.astype(int), "missing_%": miss_pct})
        miss_report = miss_report.sort_values("missing_%", ascending=False)
        print("\nApprox. missingness (first ~10M rows):")
        print(miss_report.head(50))

# ------------ Main runner ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default=r"E:\\RUL\\3\\final_features.csv", help="Path to CSV")
    ap.add_argument("--limit", type=int, default=0, help="Read only first N rows (0 = all)")
    ap.add_argument("--corr_topk", type=int, default=20, help="Top K correlation pairs to show")
    ap.add_argument("--corr_thresh", type=float, default=0.95, help="High-correlation threshold")
    ap.add_argument("--chunked", action="store_true", help="Use chunked mode for very large files")
    args = ap.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    # If file looks massive (> 4GB), prefer chunked unless user forced full read
    try:
        fsize = os.path.getsize(path)
        print(f"File size: {sizeof_fmt(fsize)}")
        prefer_chunked = fsize > 4 * 1024**3
    except Exception:
        prefer_chunked = False

    if args.chunked or prefer_chunked:
        chunked_overview(path)
        return

    nrows = args.limit if args.limit and args.limit > 0 else None
    df = load_csv(path, nrows=nrows)

    # Try to parse likely time columns before summaries
    guesses = guess_special_columns(df.columns.tolist())
    parsed_times = parse_time_columns(df, guesses.get("time_like", []))

    basic_info(df)

    print("\n=== GUESSED SPECIAL COLUMNS ===")
    for k, v in guesses.items():
        print(f"{k}: {v}")
    if parsed_times:
        print(f"Parsed as datetime: {parsed_times}")

    print("\n=== HEAD / TAIL / SAMPLE ===")
    print("HEAD:")
    print(df.head(5))
    print("\nTAIL:")
    print(df.tail(5))
    print("\nRANDOM SAMPLE:")
    print(df.sample(min(5, len(df)), random_state=42))

    print("\n=== MISSINGNESS REPORT (top 50) ===")
    miss = missingness_report(df)
    print(miss.head(50))

    zero_var, near_const = constant_columns(df)
    print("\n=== ZERO-VARIANCE COLUMNS ===")
    print(zero_var if zero_var else "(none)")
    print("\n=== NEAR-CONSTANT COLUMNS ===")
    print(near_const if near_const else "(none)")

    duplicates_check(df)

    quick_column_summaries(df)

    correlation_peek(df, top_k=args.corr_topk, thresh=args.corr_thresh)

    # Per-group overview if plausible grouping columns exist
    per_group_overview(df, guesses.get("group_like", []))

    # Class balance / target assessment if target-like columns exist
    class_balance(df, guesses.get("target_like", []))

    print("\n=== DONE ===")

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.width', 180)
    main()
