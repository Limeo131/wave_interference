"""
scripts/run_fig04_v1_boundary.py
================================
Re-run the 12 FN boundary cases at 1000 days to get A_max over full integration.
Then update the v1 data by merging with existing fig04_Amax_sweep_300d.npz.

These 12 cases burst only after ~300 days (identified by v3long 1000-day runs)
but have A_max below threshold within 300 days.

Output: output/data/hm_model/fig04_Amax_v1_1000d.npz
  Contains the full v1 A_max map (300d for most points, 1000d for the 12 boundary cases).
"""

import sys
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76

# ── configuration ───────────────────────────────────────────────────────────
S     = 2.0
DZ    = 1000.0
IMAX  = 71
DT    = 360.0 * 15.0
ALPHA = False
N_DAYS_LONG = 1000
DIAG_Z_IDX  = 22  # z = 32 km

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")

# The 12 FN cases (hb, tau) identified from fig04 analysis
FN_CASES = [
    (32, 33),
    (33, 42),
    (33, 52),
    (33, 53),
    (33, 54),
    (33, 55),
    (33, 56),
    (33, 57),
    (33, 58),
    (33, 59),
    (33, 60),
    (33, 61),
]

# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Re-running {len(FN_CASES)} boundary cases at {N_DAYS_LONG} days")
    print(f"Cases: {FN_CASES}")
    print()

    # Load existing 300d data
    d300 = np.load(os.path.join(DATA_DIR, "fig04_Amax_sweep_300d.npz"))
    hb_values = d300["hb_values"]
    tau_values = d300["tau_values"]
    amax_32_300d = d300["amax_32km"].copy()

    hb_idx = {v: i for i, v in enumerate(hb_values)}
    tau_idx = {v: j for j, v in enumerate(tau_values)}

    # Run each FN case at 1000 days
    amax_1000d = {}
    t0 = time.time()

    for case_num, (hb, tau_d) in enumerate(FN_CASES, start=1):
        print(f"  [{case_num}/{len(FN_CASES)}] hb={hb}, tau={tau_d} ... ", end="", flush=True)

        out = run_hm76(
            float(hb), float(tau_d) * 86400.0,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS_LONG, alpha_on=ALPHA,
            wave_mean_feedback=True, mean_flow_coupling=1.0,
            verbose=False,
        )

        psi_abs = np.abs(out["psi_time"][DIAG_Z_IDX, :])
        amax_val = float(np.nanmax(psi_abs))
        tpeak = int(np.nanargmax(psi_abs))

        # Also check burst time in 1000d run
        u32 = out["ud"][DIAG_Z_IDX, :]
        below_0 = np.where(u32[50:] < 0.0)[0]
        t_burst = (int(below_0[0]) + 50) if len(below_0) > 0 else -1

        amax_1000d[(hb, tau_d)] = {
            "amax": amax_val,
            "tpeak": tpeak,
            "t_burst": t_burst,
            "amax_300d": amax_32_300d[hb_idx[hb], tau_idx[tau_d]],
        }

        print(f"A_max={amax_val:.4e} (was {amax_32_300d[hb_idx[hb], tau_idx[tau_d]]:.4e}), "
              f"t_peak={tpeak}, t_burst={t_burst}")

    elapsed = time.time() - t0
    print(f"\nAll {len(FN_CASES)} cases done in {elapsed:.0f}s")

    # Build updated v1 map: use 1000d values for boundary cases
    amax_v1 = amax_32_300d.copy()
    for (hb, tau_d), info in amax_1000d.items():
        i = hb_idx[hb]
        j = tau_idx[tau_d]
        amax_v1[i, j] = info["amax"]

    # Save
    out_path = os.path.join(DATA_DIR, "fig04_Amax_v1_1000d.npz")
    np.savez_compressed(
        out_path,
        hb_values=hb_values,
        tau_values=tau_values,
        amax_32km=amax_v1,
        amax_32km_300d=amax_32_300d,
        fn_cases=np.array(FN_CASES),
        fn_amax_1000d=np.array([amax_1000d[tuple(c)]["amax"] for c in FN_CASES]),
        fn_tpeak_1000d=np.array([amax_1000d[tuple(c)]["tpeak"] for c in FN_CASES]),
        fn_tburst_1000d=np.array([amax_1000d[tuple(c)]["t_burst"] for c in FN_CASES]),
        n_days_long=np.array(N_DAYS_LONG),
    )
    print(f"\nSaved: {out_path}")
    print("Done.")
