"""
scripts/run_sweep_timing.py
============================
Parallel sweep computing t_A (first day |psi(32km)| > A_c) and auxiliary
timing diagnostics for all 2501 (hb, tau) cases.

Also computes:
  - A(t_open): wave amplitude at the moment the barrier opens
  - M(t_A): barrier metric at the moment A_c is first crossed

Reuses t_open from hb_tau_Mmax_sweep.npz for cross-validation.

Output: output/data/hm_model/Acrossing_barrier_timing.npz
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76, compute_m2, compute_min_m2

# ── Configuration (same as production sweep) ────────────────────────────────
S = 2.0
DZ = 1000.0
IMAX = 71
DT = 360.0 * 15.0
ALPHA_ON = False
WAVE_MEAN_FEEDBACK = True
MEAN_FLOW_COUPLING = 1.0
N_DAYS = 300

# Propagation-barrier settings
Z1_KM = 25.0
Z2_KM = 40.0
PHASE_SPEED_C = 0.0

# Wave-amplitude settings
Z_A_IDX = 22  # z = 32 km
A_C = 5.76e6  # critical amplitude threshold (m^2/s)

# Grid
HB_VALUES = np.arange(11, 52, 1, dtype=float)
TAU_VALUES = np.arange(1, 62, 1, dtype=float)

N_WORKERS = 32

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")


def _run_one(args):
    """Run one case and compute timing diagnostics."""
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

        # Wave amplitude at 32 km: |psi(z=32km, t)|
        psi = out['psi_time'][Z_A_IDX, :]  # complex, shape (n_days,)
        A_t = np.abs(psi)  # daily amplitude time series

        # Propagation-barrier metric M(t)
        m2_out = dict(ud=out['ud'], betae_time=out['betae_time'], eps=out['eps'])
        m2 = compute_m2(m2_out, c=PHASE_SPEED_C, s=S)
        M_t = compute_min_m2(m2, out['z'], z1_km=Z1_KM, z2_km=Z2_KM)

        # t_A: first day A(t) > A_c
        above_Ac = np.where(A_t > A_C)[0]
        t_A = int(above_Ac[0]) if len(above_Ac) > 0 else -1

        # t_open: first day M(t) > 0
        above_zero = np.where(M_t > 0)[0]
        t_open = int(above_zero[0]) if len(above_zero) > 0 else -1

        # A(t_open): amplitude when barrier opens
        A_at_open = float(A_t[t_open]) if t_open >= 0 else np.nan

        # M(t_A): barrier state when A_c is first crossed
        M_at_Across = float(M_t[t_A]) if t_A >= 0 else np.nan

        # Burst diagnostic
        min_u32 = float(np.nanmin(out['ud'][Z_A_IDX, 50:]))
        burst = int(min_u32 < 0)

        return hb, tau_d, t_A, t_open, A_at_open, M_at_Across, burst

    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        return hb, tau_d, -1, -1, np.nan, np.nan, -1


def main():
    t0 = time.time()
    n_hb = len(HB_VALUES)
    n_tau = len(TAU_VALUES)
    total = n_hb * n_tau
    print(f"Running timing sweep: {n_hb} x {n_tau} = {total} cases, {N_WORKERS} workers")

    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(_run_one, tasks)

    # Organize
    t_A_map = np.full((n_hb, n_tau), np.nan)
    t_open_map = np.full((n_hb, n_tau), np.nan)
    A_at_open_map = np.full((n_hb, n_tau), np.nan)
    M_at_Across_map = np.full((n_hb, n_tau), np.nan)
    burst_map = np.full((n_hb, n_tau), -1, dtype=int)

    for hb, tau_d, t_A, t_open, A_at_open, M_at_Across, burst in results:
        i = int(hb - HB_VALUES[0])
        j = int(tau_d - TAU_VALUES[0])
        t_A_map[i, j] = t_A if t_A >= 0 else np.nan
        t_open_map[i, j] = t_open if t_open >= 0 else np.nan
        A_at_open_map[i, j] = A_at_open
        M_at_Across_map[i, j] = M_at_Across
        burst_map[i, j] = burst

    delta_t_map = t_open_map - t_A_map

    # Save
    outpath = os.path.join(DATA_DIR, "Acrossing_barrier_timing.npz")
    np.savez(outpath,
             hb_values=HB_VALUES,
             tau_values=TAU_VALUES,
             burst_300_map=burst_map,
             t_A_map=t_A_map,
             t_open_map=t_open_map,
             delta_t_map=delta_t_map,
             A_at_open_map=A_at_open_map,
             M_at_Across_map=M_at_Across_map,
             A_c=A_C,
             z_A_km=32.0,
             z1_barrier_km=Z1_KM,
             z2_barrier_km=Z2_KM,
             )

    elapsed = time.time() - t0

    # Quick summary
    burst_mask = (burst_map == 1)
    n_burst = np.sum(burst_mask)
    dt_burst = delta_t_map[burst_mask]
    dt_valid = dt_burst[~np.isnan(dt_burst)]

    print(f"\nDone in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"Saved to: {outpath}")
    print(f"\nBurst cases: {n_burst}")
    print(f"  t_A < t_open: {np.sum(dt_valid > 0)}")
    print(f"  t_A = t_open: {np.sum(dt_valid == 0)}")
    print(f"  t_A > t_open: {np.sum(dt_valid < 0)}")
    print(f"  Median Δt: {np.median(dt_valid):.1f} days")
    print(f"  Range: [{dt_valid.min():.0f}, {dt_valid.max():.0f}] days")


if __name__ == "__main__":
    main()
