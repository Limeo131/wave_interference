"""
scripts/run_sweep_boundary_long.py
===================================
Re-run boundary-region cases from hb_tau_sweep_v3 with n_days=1000
to verify burst/no-burst classification near the transition boundary.

For each tau row, takes ±3 hb grid points around the first burst index.
Results cached as sweep_v3long_hb{hb}_tau{tau}.npz
Summary saved as hb_tau_sweep_v3long.npz (same shape as v3, but
only boundary cells are updated; interior cells copied from v3).

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/run_sweep_boundary_long.py
"""

import sys, os, time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hm76 import run_hm76

S, DZ, IMAX, DT, ALPHA = 2.0, 1000.0, 71, 360.0*15.0, False
N_DAYS_LONG  = 1000
BURST_Z_IDX  = 22      # ~32 km
BURST_THRESH = 0.0
SPINUP_DAYS  = 50
N_WORKERS    = 32

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)

# Load v3 sweep
v3 = np.load(os.path.join(DATA_DIR, "hb_tau_sweep_v3.npz"))
burst_map_v3   = v3["burst_map"].copy()
min_u32_map_v3 = v3["min_u32_map"].copy()
hb_values      = v3["hb_values"]
tau_values     = v3["tau_values"]

# Identify boundary cases (±3 around first burst per tau row)
boundary_set = set()
for j, tau in enumerate(tau_values):
    col = burst_map_v3[:, j]
    burst_idxs = np.where(col == 1)[0]
    if len(burst_idxs) == 0 or len(burst_idxs) == len(col):
        continue
    first_burst = burst_idxs[0]
    for di in range(-3, 4):
        ii = first_burst + di
        if 0 <= ii < len(hb_values):
            boundary_set.add((int(ii), int(j)))

tasks = [(i, j, hb_values[i], tau_values[j]) for (i, j) in sorted(boundary_set)]
print(f"Boundary cases: {len(tasks)}  (n_days={N_DAYS_LONG}, workers={N_WORKERS})")
print(f"Estimated time: {len(tasks)*10/N_WORKERS/60:.1f} min")


def _run_one(args):
    i, j, hb, tau_d = args
    name = f"sweep_v3long_hb{hb:.0f}_tau{tau_d:.2f}"
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(path):
        dd = np.load(path)
        return i, j, float(dd["min_u32"]), int(dd["burst"])
    try:
        out = run_hm76(
            hb, tau_d * 86400.,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS_LONG, alpha_on=ALPHA,
            wave_mean_feedback=True, mean_flow_coupling=1.0,
            verbose=False,
        )
        u32   = out["ud"][BURST_Z_IDX, :]
        min_u = float(np.nanmin(u32[SPINUP_DAYS:]))
        burst = int(min_u < BURST_THRESH)
        np.savez_compressed(path, min_u32=np.array(min_u), burst=np.array(burst))
        return i, j, min_u, burst
    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        return i, j, np.nan, -1


if __name__ == "__main__":
    t0 = time.time()

    burst_map_long   = burst_map_v3.copy()
    min_u32_map_long = min_u32_map_v3.copy()

    changed = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for k, (i, j, min_u, burst) in enumerate(
            pool.imap_unordered(_run_one, tasks), start=1
        ):
            if burst != burst_map_v3[i, j]:
                print(f"  CHANGED hb={hb_values[i]:.0f} tau={tau_values[j]:.0f}: "
                      f"v3={'B' if burst_map_v3[i,j] else '.'} -> "
                      f"long={'B' if burst else '.'} (min_u={min_u:.1f})")
                changed += 1
            burst_map_long[i, j]   = burst
            min_u32_map_long[i, j] = min_u
            if k % 50 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} done  ({time.time()-t0:.0f}s)")

    print(f"\nFinished in {time.time()-t0:.1f}s.  Changed cells: {changed}/{len(tasks)}")

    summary_path = os.path.join(DATA_DIR, "hb_tau_sweep_v3long.npz")
    np.savez_compressed(
        summary_path,
        burst_map=burst_map_long,
        min_u32_map=min_u32_map_long,
        hb_values=hb_values,
        tau_values=tau_values,
    )
    print(f"Saved: {summary_path}")
