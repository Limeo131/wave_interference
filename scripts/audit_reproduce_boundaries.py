"""
scripts/audit_reproduce_boundaries.py
=====================================
STEP 1 of the second-order-boundary feasibility audit.

Reproduce and cross-check the two reference boundaries used in the manuscript:
  (i)  wind-reversal transition boundary in (hb, tau)   [Fig 1 / Fig 6 dashed]
  (ii) full nonlinear M_max = 0 barrier-opening boundary [Fig 6a green]

We do NOT modify or overwrite any existing script or figure.

Checks performed:
  1. Load hb_tau_mechanism_1000d.npz (canonical Fig 1/6 source) and
     hb_tau_Mmax_sweep.npz (300-day M_max sweep) and hb_tau_sweep_v3long.npz.
  2. Confirm grid = 41 x 61 (hb 11..51, tau 1..61).
  3. Confirm burst_1000 has 1301 transition / 1200 no-transition.
  4. Confirm Mmax_1000_legacy>0 and Amax_1000_legacy>=Ac each classify
     burst_1000 at 2501/2501 (the figure's stated claim).
  5. Recompute the wind-reversal boundary and the Mmax=0 boundary curves
     exactly as make_manuscript_fig01/06 do, and save them.
  6. Independently re-run a handful of grid cells with the CURRENT run_hm76
     at n_days=300 and confirm min_u32 / burst / Mmax(300) reproduce the
     stored 300-day sweep values -> proves the current code reproduces the
     saved sweep.

Output:
  output/data/hm_model/audit_reference_boundaries.npz
"""
import os, sys, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hm76 import run_hm76, compute_m2, compute_min_m2

DATA = os.path.join(ROOT, "output", "data", "hm_model")


def transition_boundary(burst_map, hb_values, tau_values):
    """First-transition midpoint hb for each tau (matches make_manuscript_fig01/06)."""
    bhb, btau = [], []
    for j, tau in enumerate(tau_values):
        col = burst_map[:, j]
        idx = np.where(col == 1)[0]
        if len(idx) > 0:
            first = idx[0]
            if first > 0:
                bhb.append(0.5 * (hb_values[first - 1] + hb_values[first]))
            else:
                bhb.append(hb_values[0] - 0.5)
            btau.append(tau)
    return np.array(bhb), np.array(btau)


def contour_zero_boundary(field, hb_values, tau_values):
    """
    First hb (per tau) at which `field` crosses from <0 to >=0, with linear
    interpolation in hb. Approximates the field=0 contour as a boundary curve.
    """
    bhb, btau = [], []
    for j, tau in enumerate(tau_values):
        col = field[:, j]
        # find first index where col >= 0
        pos = np.where(col >= 0)[0]
        if len(pos) == 0:
            continue
        first = pos[0]
        if first == 0:
            bhb.append(hb_values[0] - 0.5)
        else:
            v0, v1 = col[first - 1], col[first]
            if v1 == v0:
                hbc = hb_values[first]
            else:
                # linear interp where field crosses 0
                frac = (0.0 - v0) / (v1 - v0)
                hbc = hb_values[first - 1] + frac * (hb_values[first] - hb_values[first - 1])
            bhb.append(hbc)
        btau.append(tau)
    return np.array(bhb), np.array(btau)


def main():
    print("=" * 70)
    print("STEP 1: reproduce reference boundaries")
    print("=" * 70)

    mech = np.load(os.path.join(DATA, "hb_tau_mechanism_1000d.npz"))
    mmax_sweep = np.load(os.path.join(DATA, "hb_tau_Mmax_sweep.npz"))
    v3 = np.load(os.path.join(DATA, "hb_tau_sweep_v3long.npz"))

    hb = mech["hb_values"]; tau = mech["tau_values"]
    print(f"grid: hb {hb.min():.0f}..{hb.max():.0f} ({len(hb)}), "
          f"tau {tau.min():.0f}..{tau.max():.0f} ({len(tau)})")
    assert len(hb) == 41 and len(tau) == 61

    burst_1000 = mech["burst_1000"]
    Mmax_1000 = mech["Mmax_1000_legacy"]
    Amax_1000 = mech["Amax_1000_legacy"]
    A_c = float(mech["A_c"])
    print(f"A_c = {A_c:.4e}  z_A = {float(mech['z_A_km']):.1f} km  "
          f"barrier layer {float(mech['z1_barrier_km']):.0f}-{float(mech['z2_barrier_km']):.0f} km  "
          f"c_barrier={float(mech['c_barrier']):.2f}")

    n_tr = int(np.sum(burst_1000 == 1)); n_no = int(np.sum(burst_1000 == 0))
    print(f"burst_1000: {n_tr} transition / {n_no} no-transition  (expect 1301/1200)")
    assert n_tr == 1301 and n_no == 1200

    agr_M = int(np.sum((Mmax_1000 > 0).astype(int) == burst_1000))
    agr_A = int(np.sum((Amax_1000 >= A_c).astype(int) == burst_1000))
    print(f"Mmax_1000>0 == burst_1000 : {agr_M}/2501")
    print(f"Amax_1000>=Ac == burst_1000: {agr_A}/2501")

    # consistency of v3long with mechanism file
    print(f"v3long burst_map == mech burst_1000 : "
          f"{int(np.sum(v3['burst_map']==burst_1000))}/2501")

    # 300-day sweep cross-check
    burst_300 = mech["burst_300"]
    print(f"300-day burst_300: {int(np.sum(burst_300==1))} transition "
          f"(expect 1289 per steering)")
    print(f"Mmax_sweep.burst_map == mech burst_300: "
          f"{int(np.sum(mmax_sweep['burst_map']==burst_300))}/2501")

    # --- boundaries ---
    bhb_tr, btau_tr = transition_boundary(burst_1000, hb, tau)
    bhb_M, btau_M = contour_zero_boundary(Mmax_1000, hb, tau)
    print(f"\nTransition boundary: {len(bhb_tr)} tau points, "
          f"hb range {bhb_tr.min():.1f}..{bhb_tr.max():.1f}")
    print(f"Mmax=0 boundary:     {len(bhb_M)} tau points, "
          f"hb range {bhb_M.min():.1f}..{bhb_M.max():.1f}")

    # --- reproduce a few cells with current code (300-day, matches Mmax sweep) ---
    print("\n--- Independent re-run of sample cells (n_days=300, current run_hm76) ---")
    S, DZ, IMAX, DT = 2.0, 1000.0, 71, 360.0 * 15.0
    test_cells = [(30.0, 10.0), (30.0, 20.0), (20.0, 5.0), (45.0, 40.0),
                  (11.0, 1.0), (51.0, 61.0)]
    repro = []
    for hbv, tauv in test_cells:
        out = run_hm76(hbv, tauv * 86400.0, s=S, dz=DZ, imax=IMAX, dt=DT,
                       n_days=300, alpha_on=False, wave_mean_feedback=True,
                       mean_flow_coupling=1.0)
        u32 = out["ud"][22, :]
        min_u32 = float(np.nanmin(u32[50:]))
        burst = int(min_u32 < 0.0)
        m2 = compute_m2(dict(ud=out["ud"], betae_time=out["betae_time"], eps=out["eps"]),
                        c=0.0, s=S)
        Mt = compute_min_m2(m2, out["z"], z1_km=25.0, z2_km=40.0)
        Mmax = float(np.nanmax(Mt))
        i = int(hbv - hb[0]); j = int(tauv - tau[0])
        stored_min = float(mmax_sweep["min_u32_map"][i, j])
        stored_burst = int(mmax_sweep["burst_map"][i, j])
        stored_Mmax = float(mmax_sweep["Mmax_map"][i, j])
        du = abs(min_u32 - stored_min)
        dM = abs(Mmax - stored_Mmax)
        ok = (burst == stored_burst) and (du < 1e-6) and (dM < 1e-15)
        repro.append((hbv, tauv, min_u32, stored_min, du, Mmax, stored_Mmax, dM,
                      burst, stored_burst, ok))
        print(f"  hb={hbv:4.0f} tau={tauv:4.0f}: min_u32 {min_u32:+8.4f} vs "
              f"{stored_min:+8.4f} (d={du:.2e}); Mmax {Mmax:+.4e} vs "
              f"{stored_Mmax:+.4e} (d={dM:.2e}); burst {burst}=={stored_burst} "
              f"-> {'OK' if ok else 'MISMATCH'}")

    all_ok = all(r[-1] for r in repro)
    print(f"\nAll sample cells reproduce stored sweep: {all_ok}")

    np.savez(os.path.join(DATA, "audit_reference_boundaries.npz"),
             hb_values=hb, tau_values=tau,
             burst_1000=burst_1000, Mmax_1000=Mmax_1000, Amax_1000=Amax_1000,
             A_c=A_c,
             boundary_transition_hb=bhb_tr, boundary_transition_tau=btau_tr,
             boundary_Mmax0_hb=bhb_M, boundary_Mmax0_tau=btau_M,
             repro_cells=np.array([(r[0], r[1], r[2], r[3], r[5], r[6],
                                    r[8], r[9]) for r in repro]))
    print(f"\nSaved: {os.path.join(DATA, 'audit_reference_boundaries.npz')}")
    return all_ok


if __name__ == "__main__":
    main()
