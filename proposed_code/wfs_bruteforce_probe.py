
import argparse, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.signal import welch
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def parse_list(s, typ=int):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]

def signed_int24_from_bytes_le(b):
    """Convert little-endian 3-byte signed integers to int32 numpy array."""
    b = np.frombuffer(b, dtype=np.uint8)
    if b.size % 3 != 0:
        raise ValueError("Byte length not multiple of 3 for int24.")
    b = b.reshape(-1, 3)
    # build 32-bit little-endian from 3 bytes with sign extension
    x = (b[:,0].astype(np.int32) |
         (b[:,1].astype(np.int32) << 8) |
         (b[:,2].astype(np.int32) << 16))
    # sign extend 24th bit
    neg = (x & 0x00800000) != 0
    x[neg] |= 0xFF000000
    return x

def band_energy_ratio(x, fs, lo=80_000, hi=400_000, lowband_hi=5_000):
    if not HAVE_SCIPY:
        # fallback: simple FFT-based energy ratio
        n = min(len(x), 65536)
        X = np.fft.rfft(x[:n] * np.hanning(n))
        f = np.fft.rfftfreq(n, d=1.0/fs)
        Pxx = (np.abs(X)**2) / n
        eps = 1e-20
        band = (f >= lo) & (f <= hi)
        low  = (f <= lowband_hi)
        e_band = np.trapz(Pxx[band], f[band]) + eps
        e_low  = np.trapz(Pxx[low],  f[low])  + eps
        return float(e_band / e_low)
    else:
        f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 8192))
        eps = 1e-20
        band = (f >= lo) & (f <= hi)
        low  = (f <= lowband_hi)
        e_band = np.trapz(Pxx[band], f[band]) + eps
        e_low  = np.trapz(Pxx[low],  f[low])  + eps
        return float(e_band / e_low)

def main():
    ap = argparse.ArgumentParser(description="Bruteforce probe for proprietary .wfs-like waveform containers.")
    ap.add_argument("--path", required=True, help="Path to .wfs file")
    ap.add_argument("--fs", type=int, default=1_000_000, help="Sampling frequency (Hz), default 1 MHz")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds to peek for stats")
    ap.add_argument("--n_channels_list", type=str, default="4,6,8,10,12", help="Comma list of plausible channel counts")
    ap.add_argument("--dtypes", type=str, default="int16,int24le,int32,float32", help="Comma list of dtypes to try")
    ap.add_argument("--max_header_mb", type=float, default=64.0, help="Maximum plausible single header size (MB)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[ERR] File not found: {p}", file=sys.stderr); sys.exit(1)

    file_size = p.stat().st_size
    print(f"[INFO] File: {p.name} | Size: {file_size:,} bytes ({file_size/1e9:.3f} GB)")

    n_channels_list = parse_list(args.n_channels_list, int)
    dtype_names = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    max_header_bytes = int(args.max_header_mb * 1024 * 1024)
    fs = args.fs

    def itemsize_of(dt):
        if dt == "int24le": return 3
        return np.dtype(dt).itemsize

    def duration_score(dur):
        # Prefer durations between ~1,000 s and 20,000 s with a soft peak near ~6,900 s
        if dur <= 300 or dur > 24*3600:
            return -10.0
        # bell around 6,900
        return -((dur - 6888.0)/2000.0)**2

    candidates = []
    # Try to infer a single pre-header by modulus; otherwise brute-force common sizes
    common_headers = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                      131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608]
    for dt in dtype_names:
        isz = itemsize_of(dt)
        for nc in n_channels_list:
            denom = isz * nc
            # exact remainder trick
            rem = file_size % denom
            if rem <= max_header_bytes and (file_size - rem) > denom:
                n_samples = (file_size - rem)//denom
                dur = n_samples / fs
                score = 2.0  # base
                if dt == "int16": score += 1.5
                score += duration_score(dur)
                candidates.append((dt, nc, int(rem), int(n_samples), dur, score, "modulus-fit"))
            # test common header sizes
            for hb in common_headers:
                if hb > max_header_bytes or hb >= file_size: 
                    continue
                payload = file_size - hb
                if payload % denom == 0:
                    n_samples = payload // denom
                    if n_samples <= 0: 
                        continue
                    dur = n_samples / fs
                    score = 1.0
                    if dt == "int16": score += 1.0
                    score += duration_score(dur) - math.log2(hb+1)*0.01
                    candidates.append((dt, nc, int(hb), int(n_samples), dur, score, "common-hdr"))

    if not candidates:
        print("[ERROR] No plausible (dtype, channels, header) found within constraints.")
        print("Try broadening --n_channels_list or increasing --max_header_mb, or use vendor software to export.")
        sys.exit(2)

    # Sort by score
    candidates.sort(key=lambda x: (-x[5], x[2]))
    print("[INFO] Top candidate hypotheses:")
    for i, (dt, nc, hb, nsmpl, dur, score, kind) in enumerate(candidates[:8], 1):
        print(f"  {i:>2}. dtype={dt:8s}  ch={nc:2d}  header={hb} bytes  samples/ch={nsmpl:,}  ~{dur:.2f}s   ({kind}, score={score:.2f})")

    # Use the top hypothesis
    dt, nc, hb, nsmpl, dur, score, kind = candidates[0]
    print(f"[INFO] Using best hypothesis → dtype={dt}, channels={nc}, header={hb}, duration≈{dur:.2f}s")

    # Peek a few seconds
    seconds = min(args.seconds, dur)
    smpl = int(seconds * fs)
    if dt == "int24le":
        # read raw bytes then convert
        with open(p, "rb") as f:
            f.seek(hb, 0)
            # bytes needed per sample frame
            bpf = 3 * nc
            need = smpl * bpf
            raw = f.read(need)
            if len(raw) < need:
                print("[WARN] Could not read requested bytes; truncating.")
            arr32 = signed_int24_from_bytes_le(raw)
            # reshape [frames, channels]
            frames = arr32.size // nc
            arr = arr32[:frames*nc].reshape(frames, nc).astype(np.float32)
    else:
        # memmap then slice
        raw = np.memmap(p, dtype=dt, mode="r", offset=hb, shape=(nsmpl*nc,))
        arr = np.array(raw[:smpl*nc], dtype=np.float32).reshape(-1, nc)
        del raw

    # quick stats
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0) + 1e-12
    kurt = (np.mean((arr - mu)**4, axis=0) / (sd**2))

    # spectral ratios per channel
    ratios = [band_energy_ratio(arr[:,ch], fs) for ch in range(nc)]
    labels = ["AE-like" if r > 0.3 else "low-freq/pressure-like" for r in ratios]

    df = pd.DataFrame({
        "channel": np.arange(1, nc+1),
        "mean": mu,
        "std": sd,
        "kurtosis": kurt,
        "AE_band_ratio(100-400k / 0-5k)": ratios,
        "auto_label": labels
    })
    outdir = Path("wfs_probe_out"); outdir.mkdir(exist_ok=True, parents=True)
    csv_path = outdir / f"{p.stem}_probe_summary.csv"
    df.to_csv(csv_path, index=False)

    # save tiny waveform plots for first 0.01 s
    try:
        import matplotlib.pyplot as plt
        snippet_samples = min(int(0.01*fs), arr.shape[0])
        t = np.arange(snippet_samples)/fs
        for ch in range(nc):
            plt.figure()
            plt.plot(t, arr[:snippet_samples, ch])
            plt.xlabel("Time [s]"); plt.ylabel(f"Ch {ch+1} (raw units)")
            plt.title(f"{p.stem} - Channel {ch+1} (first 0.01 s)")
            plt.tight_layout()
            fig_path = outdir / f"{p.stem}_ch{ch+1}_snippet.png"
            plt.savefig(fig_path, dpi=140); plt.close()
        print(f"[OK] Saved probe summary → {csv_path}")
        print(f"[OK] Saved tiny snippet plots in {outdir}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    n_ae = sum(1 for x in labels if "AE-like" in x)
    print(f"[SUMMARY] Hypothesis gave AE-like: {n_ae} / {nc} channels")
    print(df)

    print("\nIf this looks wrong (e.g., all zeros, NaNs, or crazy values), "
          "the container probably has interleaved block headers (vendor-proprietary). "
          "In that case, best path is to export waveforms to CSV/MAT/HDF5 using the vendor software.")
    print("If you're using PAC/Mistras/Noesis tools, look for 'Export Waveforms' or 'Save As' to open formats.")

if __name__ == "__main__":
    main()
