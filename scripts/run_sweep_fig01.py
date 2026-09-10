"""
scripts/run_sweep_fig01.py
==========================
Parallel (hb, tau) sweep for fig01_hb_tau_phase_diagram.

Grid (matching reference figure style):
  hb  : 11, 13, 15, ..., 51 m   (step 2,  21 values)
  tau : 1,  4,  7, ..., 61 days (step 3,  21 values)

Uses multiprocessing.Pool to run ~32 workers in parallel.
Results are cached to output/data/hm_model/sweep_v2_hb{hb}_tau{tau}.npz
Summary is saved to output/data/hm_model/hb_tau_sweep_v2.npz

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/run_sweep_fig01.py
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76

# ── configuration ───────────────────────────────────────────────────────────
S     = 2.0
DZ    = 1000.0
IMAX  = 71
DT    = 360.0 * 15.0
ALPHA = False          # alpha_on=False (matches existing sweep)
N_DAYS = 300

# Burst criterion: u at ~32 km (index 22) drops below 0 m/s after spin-up
BURST_Z_IDX  = 22      # z ≈ 32 km
BURST_THRESH = 0.0     # m/s
SPINUP_DAYS  = 50      # skip first 50 days

# New grid
HB_VALUES  = np.arange(11, 52, 1, dtype=float)   # 11,12,...,51  (41 values)
TAU_VALUES = np.arange(1,  62, 1, dtype=float)   # 1, 2, 3,...,61 (61 values)

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)

N_WORKERS = 32

# ── per-run function (must be top-level for pickle) ─────────────────────────
def _run_one(args):
    hb, tau_d = args
    name = f"sweep_v3_hb{hb:.0f}_tau{tau_d:.2f}"
    path = os.path.join(DATA_DIR, f"{name}.npz")

    if os.path.exists(path):
        d = np.load(path)
        return hb, tau_d, float(d["min_u32"]), int(d["burst"])

    try:
        out = run_hm76(
            hb, tau_d * 86400.0,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS, alpha_on=ALPHA,
            wave_mean_feedback=True, mean_flow_coupling=1.0,
            verbose=False,
        )
        u32   = out["ud"][BURST_Z_IDX, :]
        min_u = float(np.nanmin(u32[SPINUP_DAYS:]))
        burst = int(min_u < BURST_THRESH)
        np.savez_compressed(path, min_u32=np.array(min_u), burst=np.array(burst))
        return hb, tau_d, min_u, burst
    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        return hb, tau_d, np.nan, -1


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]
    NH, NT = len(HB_VALUES), len(TAU_VALUES)
    total  = len(tasks)

    print(f"Grid: {NH} hb values × {NT} tau values = {total} runs")
    print(f"Workers: {N_WORKERS}")
    print(f"Estimated time (new runs only): ~{total * 10 / N_WORKERS / 60:.1f} min")
    print()

    t0 = time.time()

    burst_map   = np.full((NH, NT), -1,  dtype=int)
    min_u32_map = np.full((NH, NT), np.nan)

    hb_idx  = {v: i for i, v in enumerate(HB_VALUES)}
    tau_idx = {v: j for j, v in enumerate(TAU_VALUES)}

    with mp.Pool(processes=N_WORKERS) as pool:
        for k, (hb, tau_d, min_u, burst) in enumerate(
            pool.imap_unordered(_run_one, tasks), start=1
        ):
            i = hb_idx[hb]
            j = tau_idx[tau_d]
            burst_map[i, j]   = burst
            min_u32_map[i, j] = min_u
            if k % 20 == 0 or k == total:
                elapsed = time.time() - t0
                print(f"  {k}/{total} done  ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nAll runs finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save summary
    summary_path = os.path.join(DATA_DIR, "hb_tau_sweep_v3.npz")
    np.savez_compressed(
        summary_path,
        burst_map=burst_map,
        min_u32_map=min_u32_map,
        hb_values=HB_VALUES,
        tau_values=TAU_VALUES,
    )
    print(f"Saved summary: {summary_path}")
    print(f"  burst_map shape: {burst_map.shape}")
    print(f"  burst count: {np.sum(burst_map == 1)} / {total}")
