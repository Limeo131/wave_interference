"""
scripts/run_sweep_Mmax.py
=========================
Parallel (hb, tau) sweep computing M_max — the propagation-barrier metric.

Metric definition (locked 2026-08-16):
    M(t) = min_{z in [25,40] km} m²_stat(z,t)
    M_max(hb, tau) = max_t M(t)   over full 300-day integration

    m²_stat uses c=0 (stationary-wave refractive index).
    Classification: M_max > 0  =>  barrier removed  =>  burst predicted.

Grid:
    hb  = 11, 12, ..., 51 m     (41 values)
    tau = 1, 2, ..., 61 days    (61 values)
    Total: 2501 cases

Production config (matching run_sweep_fig01.py):
    alpha_on = False
    wave_mean_feedback = True
    mean_flow_coupling = 1.0
    n_days = 300
    s = 2.0, dz = 1000.0, imax = 71, dt = 360*15

Output:
    output/data/hm_model/hb_tau_Mmax_sweep.npz

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/run_sweep_Mmax.py
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76, compute_m2, compute_min_m2

# ── Configuration ───────────────────────────────────────────────────────────
S = 2.0
DZ = 1000.0
IMAX = 71
DT = 360.0 * 15.0
ALPHA_ON = False
WAVE_MEAN_FEEDBACK = True
MEAN_FLOW_COUPLING = 1.0
N_DAYS = 300

# Propagation-barrier metric settings
Z1_KM = 25.0
Z2_KM = 40.0
PHASE_SPEED_C = 0.0

# Burst criterion (same as production sweep)
BURST_Z_IDX = 22  # z = 32 km
BURST_THRESH = 0.0
SPINUP_DAYS = 50

# Grid
HB_VALUES = np.arange(11, 52, 1, dtype=float)   # 41 values
TAU_VALUES = np.arange(1, 62, 1, dtype=float)   # 61 values

N_WORKERS = 32

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)


def _run_one(args):
    """Run one (hb, tau) case and return scalar diagnostics."""
    hb, tau_d = args
    try:
        out = run_hm76(
            hb, tau_d * 86400.0,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS, alpha_on=ALPHA_ON,
            wave_mean_feedback=WAVE_MEAN_FEEDBACK,
            mean_flow_coupling=MEAN_FLOW_COUPLING,
            verbose=False,
        )

        # Burst diagnostic
        u32 = out["ud"][BURST_Z_IDX, :]
        min_u32 = float(np.nanmin(u32[SPINUP_DAYS:]))
        burst = int(min_u32 < BURST_THRESH)

        # Propagation-barrier metric
        m2_out = dict(ud=out["ud"], betae_time=out["betae_time"], eps=out["eps"])
        m2 = compute_m2(m2_out, c=PHASE_SPEED_C, s=S)
        M_t = compute_min_m2(m2, out["z"], z1_km=Z1_KM, z2_km=Z2_KM)
        Mmax = float(np.nanmax(M_t))

        # Time of first barrier opening (M > 0)
        pos_idx = np.where(M_t > 0)[0]
        t_open = int(pos_idx[0]) if len(pos_idx) > 0 else -1

        return hb, tau_d, min_u32, burst, Mmax, t_open

    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        return hb, tau_d, np.nan, -1, np.nan, -1


def main():
    t0 = time.time()
    n_hb = len(HB_VALUES)
    n_tau = len(TAU_VALUES)
    total = n_hb * n_tau
    print(f"Running M_max sweep: {n_hb} x {n_tau} = {total} cases, {N_WORKERS} workers")

    # Build task list
    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]

    # Run in parallel
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_run_one, tasks)

    # Organize into arrays
    Mmax_map = np.full((n_hb, n_tau), np.nan)
    min_u32_map = np.full((n_hb, n_tau), np.nan)
    burst_map = np.full((n_hb, n_tau), -1, dtype=int)
    t_open_map = np.full((n_hb, n_tau), np.nan)

    for hb, tau_d, min_u32, burst, Mmax, t_open in results:
        i = int(hb - HB_VALUES[0])
        j = int(tau_d - TAU_VALUES[0])
        Mmax_map[i, j] = Mmax
        min_u32_map[i, j] = min_u32
        burst_map[i, j] = burst
        t_open_map[i, j] = t_open if t_open >= 0 else np.nan

    barrier_open_map = (Mmax_map > 0).astype(int)

    # Save
    outpath = os.path.join(DATA_DIR, "hb_tau_Mmax_sweep.npz")
    np.savez(outpath,
             hb_values=HB_VALUES,
             tau_values=TAU_VALUES,
             Mmax_map=Mmax_map,
             barrier_open_map=barrier_open_map,
             burst_map=burst_map,
             min_u32_map=min_u32_map,
             t_open_map=t_open_map,
             # Metadata
             z1_km=Z1_KM,
             z2_km=Z2_KM,
             phase_speed_c=PHASE_SPEED_C,
             n_days=N_DAYS,
             alpha_on=ALPHA_ON,
             wave_mean_feedback=WAVE_MEAN_FEEDBACK,
             mean_flow_coupling=MEAN_FLOW_COUPLING,
             burst_z_idx=BURST_Z_IDX,
             burst_thresh=BURST_THRESH,
             spinup_days=SPINUP_DAYS,
             )

    elapsed = time.time() - t0
    n_burst = np.sum(burst_map == 1)
    n_noburst = np.sum(burst_map == 0)
    n_open = np.sum(barrier_open_map == 1)
    print(f"\nDone in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"Saved to: {outpath}")
    print(f"Burst cases: {n_burst}, No-burst cases: {n_noburst}")
    print(f"Barrier opened: {n_open}")
    print(f"Agreement (barrier_open == burst): {np.sum(barrier_open_map == burst_map)}/{total}")


if __name__ == "__main__":
    main()
