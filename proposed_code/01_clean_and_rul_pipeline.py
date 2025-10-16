"""
Clean + Novel RUL Pipeline (Monotone Contrastive Health Index + Isotonic RUL)

Assumptions:
- CSV has columns like ch{1..8}_{rms,mean,std,peak,skew,kurtosis,crest_factor,energy,zcr,envelope_mean} and an id column `chunk_id`.
- If there is no explicit test/run column, we treat the entire file as a single run. If you have 3 tests, pass a mapping file or add a `test_id` column beforehand.

What this script does:
1) Load and CLEAN
   - Fix any numeric features misread as datetimes (coerce to float)
   - Drop zero-variance columns and rows that are entirely NaN across features
   - Infer order index from `chunk_id` (extract digits), or use row order
   - (Optional) Build `test_id`; otherwise assume single test
   - Create a proxy ground-truth RUL per test: RUL(t) = last_index - index

2) Learn a 1-D Health Indicator (HI): Monotone Contrastive Health Index (MCHI)
   - h_t = w^T x_t (linear), with regularizers:
       * Monotonicity: penalize negative differences: sum ReLU(-(h_{t+1}-h_t))
       * Smoothness: L2 on second differences of h_t
       * Contrastive: push early vs late segments apart (hinge on margin m)
   - Implemented in PyTorch (falls back to PCA if PyTorch not installed)

3) Map HI -> RUL using isotonic regression (monotone decreasing), trained on true proxy RUL.

4) Evaluate MAE and plot:
   - HI over time per test
   - RUL truth vs prediction over time
   - Predicted RUL vs HI (calibration curve)

Usage:
    python 01_clean_and_rul_pipeline.py --path "E:\\RUL\\3\\final_features.csv" --outdir "E:\\RUL\\3\\outputs"
Optional:
    --test_col test_id  (if present)
    --seed 42

"""
from __future__ import annotations
import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd

# --------------------- Utils ---------------------

def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

NUMERIC_LIKE = ("std","kurtosis","crest_factor","rms","mean","peak","skew","energy","zcr","envelope_mean")

DT64_TO_FLOAT_COLS = None  # will be detected


def read_clean_csv(path: str) -> pd.DataFrame:
    # Read without forcing datetime; let pandas infer
    df = pd.read_csv(path, low_memory=False)

    # Identify feature columns (exclude ids)
    id_cols = [c for c in df.columns if c.lower() in {"chunk_id","id","sample_id"}]

    # Fix datetimes that should be floats (our previous script converted them)
    dt_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
    mis_dt_cols = [c for c in dt_cols if any(tok in c.lower() for tok in NUMERIC_LIKE)]
    if mis_dt_cols:
        # Convert datetime -> nanoseconds from epoch -> seconds as float
        for c in mis_dt_cols:
            s = df[c].astype("datetime64[ns]")
            # If all NaT or single value 1970-01-01, treat as zeros
            if s.isna().all() or s.dropna().nunique() == 1:
                df[c] = 0.0
            else:
                ns = s.view("int64")  # nanoseconds since epoch; NaT becomes NA
                df[c] = (ns / 1e9).astype(float)

    # Coerce obvious numeric columns
    for c in df.columns:
        if any(tok in c.lower() for tok in NUMERIC_LIKE):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop all-NaN rows across feature columns
    feat_cols = [c for c in df.columns if c not in id_cols]
    df = df.dropna(axis=0, how='all', subset=feat_cols)

    # Compute zero-variance columns to drop
    zero_var = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]
    if zero_var:
        df = df.drop(columns=zero_var)
        print(f"Dropped zero-variance columns: {zero_var}")

    # Extract an order index from chunk_id if present
    if 'chunk_id' in df.columns:
        # Extract trailing digits; fallback to positional index
        def extract_idx(x):
            if pd.isna(x):
                return np.nan
            m = re.search(r"(\d+)", str(x))
            return int(m.group(1)) if m else np.nan
        df['order'] = df['chunk_id'].map(extract_idx)
        # If many NaN, use range
        if df['order'].isna().mean() > 0.2:
            df['order'] = np.arange(len(df))
    else:
        df['order'] = np.arange(len(df))

    # If no test_id, create a single test
    if 'test_id' not in df.columns:
        df['test_id'] = 'test_1'

    # Sort within test by order
    df = df.sort_values(['test_id','order']).reset_index(drop=True)

    return df


# --------------------- Health Index Learning ---------------------

def learn_hi_torch(X: np.ndarray, groups: np.ndarray, seed: int = 42, epochs: int = 2000, lr: float = 0.05, margin: float = 0.5, lam_mon: float = 5.0, lam_smooth: float = 1.0, lam_l2: float = 1e-3) -> np.ndarray:
    try:
        import torch
        torch.manual_seed(seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # One weight vector w (linear HI)
        w = torch.nn.Parameter(torch.zeros(X.shape[1], device=device))

        opt = torch.optim.Adam([w], lr=lr)

        # Build index per group for monotonic penalties
        idx_groups = []
        start = 0
        for g in np.unique(groups):
            sel = (groups == g)
            idx = np.where(sel)[0]
            idx_groups.append(idx)

        for ep in range(epochs):
            opt.zero_grad()
            h = X_t @ w  # (N,)

            # Monotonicity: penalize negative diffs within each group
            mon_loss = torch.tensor(0.0, device=device)
            smooth_loss = torch.tensor(0.0, device=device)
            for idx in idx_groups:
                if len(idx) < 2:
                    continue
                hi = h[idx]
                d1 = hi[1:] - hi[:-1]
                mon_loss = mon_loss + torch.relu(-d1).mean()
                if len(idx) >= 3:
                    d2 = hi[2:] - 2*hi[1:-1] + hi[:-2]
                    smooth_loss = smooth_loss + (d2.pow(2).mean())

            # Contrastive: early vs late within each group (push apart by margin)
            # sample first 20% vs last 20%
            contr_loss = torch.tensor(0.0, device=device)
            for idx in idx_groups:
                n = len(idx)
                if n < 10:
                    continue
                k = max(1, n // 5)
                early = h[idx[:k]]
                late = h[idx[-k:]]
                # We want late >= early + margin (increasing degradation HI)
                # hinge on (margin - (late.mean - early.mean))
                gap = late.mean() - early.mean()
                contr_loss = contr_loss + torch.relu(margin - gap)

            l2 = (w.pow(2).mean())

            loss = lam_mon*mon_loss + lam_smooth*smooth_loss + contr_loss + lam_l2*l2
            loss.backward()
            opt.step()

        with torch.no_grad():
            h = (X_t @ w).cpu().numpy()
        return h
    except Exception as e:
        warnings.warn(f"Torch path failed ({e}). Falling back to PCA-based HI.")
        from sklearn.decomposition import PCA
        comp = PCA(n_components=1).fit_transform(X)
        return comp[:,0]


def fit_isotonic(x: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    # RUL should decrease as HI increases → y decreasing in x
    iso = IsotonicRegression(increasing=False, y_min=0).fit(x, y)
    return iso


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, required=True)
    ap.add_argument('--outdir', type=str, default=None)
    ap.add_argument('--test_col', type=str, default='test_id')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(args.path), 'outputs')
    ensure_dir(outdir)

    df = read_clean_csv(args.path)

    # Save cleaned
    clean_path = os.path.join(outdir, 'final_features_clean.csv')
    df.to_csv(clean_path, index=False)
    print(f"Saved cleaned CSV -> {clean_path}")

    # Build features matrix X and groups
    id_cols = {'chunk_id','order',args.test_col}
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].values.astype(float)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Build groups and time index per group for RUL
    groups = df[args.test_col].values

    # Create proxy true RUL per group as (max_index - index)
    rul = np.zeros(len(df), dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups==g)[0]
        orders = df.loc[idx,'order'].values
        last = np.nanmax(orders)
        rul[idx] = last - orders

    # Learn HI
    h = learn_hi_torch(Xs, groups, seed=args.seed)

    # Fit isotonic RUL(h)
    iso = fit_isotonic(h, rul)
    rul_pred = iso.predict(h)

    # Evaluate
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(rul, rul_pred)
    print(f"MAE (proxy RUL): {mae:.3f}")

    # Save artifacts
    np.save(os.path.join(outdir, 'hi.npy'), h)
    np.save(os.path.join(outdir, 'rul_true.npy'), rul)
    np.save(os.path.join(outdir, 'rul_pred.npy'), rul_pred)

    # Plots
    import matplotlib.pyplot as plt

    # 1) HI over time per test (first 3 tests if many)
    plt.figure()
    shown = 0
    for g in np.unique(groups):
        idx = np.where(groups==g)[0]
        plt.plot(df.loc[idx,'order'].values, h[idx], label=f"{g}")
        shown += 1
        if shown >= 3:
            break
    plt.xlabel('order')
    plt.ylabel('Health Index (MCHI)')
    plt.title('Health Index over time (first 3 tests)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'hi_over_time.png'), dpi=160)

    # 2) RUL truth vs prediction over time (first test)
    g0 = np.unique(groups)[0]
    idx0 = np.where(groups==g0)[0]
    plt.figure()
    t0 = df.loc[idx0,'order'].values
    plt.plot(t0, rul[idx0], label='True RUL')
    plt.plot(t0, rul_pred[idx0], label='Pred RUL')
    plt.xlabel('order')
    plt.ylabel('RUL (proxy)')
    plt.title(f'RUL true vs predicted ({g0})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rul_true_vs_pred.png'), dpi=160)

    # 3) Calibration: RUL vs HI
    plt.figure()
    plt.scatter(h, rul, s=8, alpha=0.6)
    xs = np.linspace(h.min(), h.max(), 200)
    plt.plot(xs, iso.predict(xs))
    plt.xlabel('Health Index (MCHI)')
    plt.ylabel('RUL (proxy)')
    plt.title('Calibration: RUL vs HI')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rul_vs_hi.png'), dpi=160)

    print(f"Saved plots to {outdir}")

if __name__ == '__main__':
    pd.set_option('display.width', 180)
    main()
