"""
scripts/run_sweep_fig04.py
==========================
Parallel (hb, tau) sweep to compute peak total wave amplitude A_max
for Figure 4 of the interference manuscript.

For each grid point, runs HM76 for 300 days and records:
  - max|psi(z_*, t)| over the adjustment window [0, t_adj] at multiple altitudes
  - time of peak (day index)
  - burst flag (min u at 32 km < 0 after 50 days)

Grid: hb = 11..51 (step 1, 41 values), tau = 1..61 (step 1, 61 values)
Output: output/data/hm_model/fig04_Amax_sweep.npz

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/run_sweep_fig04.py
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
# Model parameters (same as run_sweep_fig01.py)
S     = 2.0
DZ    = 1000.0
IMAX  = 71
DT    = 360.0 * 15.0     # 5400 s = 16 steps/day
ALPHA = False             # alpha_on=False (matches existing sweep)
N_DAYS = 300

# Burst criterion
BURST_Z_IDX  = 22        # z ~ 32 km
BURST_THRESH = 0.0       # m/s
SPINUP_DAYS  = 50        # skip first 50 days for burst detection

# Adjustment window for A_max: days [0, T_ADJ]
# Pre-defined before inspecting classification skill.
# Physically motivated: ~3-4 free-mode periods (~40 d each).
T_ADJ = 150              # days

# Diagnostic altitudes: z = 10 + idx*1 km
# idx=15 → 25 km, idx=20 → 30 km, idx=22 → 32 km, idx=25 → 35 km, idx=30 → 40 km
DIAG_LEVELS = {
    '25km': 15,
    '30km': 20,
    '32km': 22,
    '35km': 25,
    '40km': 30,
}

# Parameter grid
HB_VALUES  = np.arange(11, 52, 1, dtype=float)   # 41 values
TAU_VALUES = np.arange(1,  62, 1, dtype=float)   # 61 values

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)

N_WORKERS = 32

# ── per-run function ────────────────────────────────────────────────────────
def _run_one(args):
    """Run one (hb, tau) case and return A_max at each diagnostic level."""
    hb, tau_d = args

    try:
        out = run_hm76(
            hb, tau_d * 86400.0,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS, alpha_on=ALPHA,
            wave_mean_feedback=True, mean_flow_coupling=1.0,
            verbose=False,
        )

        psi = out["psi_time"]    # complex64, shape (71, n_days_output)
        ud  = out["ud"]          # shape (71, n_days_output)
        n_out = psi.shape[1]

        # Adjustment window: day 0 to T_ADJ (inclusive)
        t_end = min(T_ADJ, n_out)

        # Compute A_max and t_peak at each diagnostic level
        amax_dict = {}
        tpeak_dict = {}
        for lev_name, lev_idx in DIAG_LEVELS.items():
            psi_abs = np.abs(psi[lev_idx, :t_end])
            amax_dict[lev_name] = float(np.nanmax(psi_abs))
            tpeak_dict[lev_name] = int(np.nanargmax(psi_abs))

        # Burst flag
        u32 = ud[BURST_Z_IDX, :]
        min_u = float(np.nanmin(u32[SPINUP_DAYS:]))
        burst = int(min_u < BURST_THRESH)

        return hb, tau_d, amax_dict, tpeak_dict, min_u, burst

    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        amax_dict = {k: np.nan for k in DIAG_LEVELS}
        tpeak_dict = {k: -1 for k in DIAG_LEVELS}
        return hb, tau_d, amax_dict, tpeak_dict, np.nan, -1


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]
    NH, NT = len(HB_VALUES), len(TAU_VALUES)
    total = len(tasks)

    print(f"Figure 4 A_max sweep")
    print(f"Grid: {NH} hb values x {NT} tau values = {total} runs")
    print(f"Workers: {N_WORKERS}")
    print(f"Adjustment window: 0 to {T_ADJ} days")
    print(f"Diagnostic levels: {list(DIAG_LEVELS.keys())}")
    print(f"Estimated time: ~{total * 8 / N_WORKERS / 60:.0f} min")
    print()

    t0 = time.time()

    # Allocate output arrays
    amax_maps = {k: np.full((NH, NT), np.nan) for k in DIAG_LEVELS}
    tpeak_maps = {k: np.full((NH, NT), -1, dtype=int) for k in DIAG_LEVELS}
    burst_map = np.full((NH, NT), -1, dtype=int)
    min_u32_map = np.full((NH, NT), np.nan)

    hb_idx  = {v: i for i, v in enumerate(HB_VALUES)}
    tau_idx = {v: j for j, v in enumerate(TAU_VALUES)}

    with mp.Pool(processes=N_WORKERS) as pool:
        for count, (hb, tau_d, amax_d, tpeak_d, min_u, burst) in enumerate(
            pool.imap_unordered(_run_one, tasks), start=1
        ):
            i = hb_idx[hb]
            j = tau_idx[tau_d]
            for k in DIAG_LEVELS:
                amax_maps[k][i, j] = amax_d[k]
                tpeak_maps[k][i, j] = tpeak_d[k]
            burst_map[i, j] = burst
            min_u32_map[i, j] = min_u

            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate if rate > 0 else 0
                print(f"  {count}/{total} done  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nAll runs finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save summary
    save_dict = dict(
        hb_values=HB_VALUES,
        tau_values=TAU_VALUES,
        burst_map=burst_map,
        min_u32_map=min_u32_map,
        t_adj=np.array(T_ADJ),
        diag_levels=np.array(list(DIAG_LEVELS.keys())),
        diag_indices=np.array(list(DIAG_LEVELS.values())),
    )
    for k in DIAG_LEVELS:
        save_dict[f'amax_{k}'] = amax_maps[k]
        save_dict[f'tpeak_{k}'] = tpeak_maps[k]

    out_path = os.path.join(DATA_DIR, "fig04_Amax_sweep.npz")
    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"  Arrays: {list(save_dict.keys())}")
    print(f"  Burst count: {np.sum(burst_map == 1)} / {total}")
    print(f"  A_max at 32km — min: {np.nanmin(amax_maps['32km']):.4f}, "
          f"max: {np.nanmax(amax_maps['32km']):.4f}")
