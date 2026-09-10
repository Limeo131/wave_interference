"""
scripts/run_second_order_sweep.py
=================================
STEPS 4-5: compute the second-order predicted transition boundary
hb_c^pred(tau) for every tau in the manuscript grid and for four analysis
windows (150, 250, 500, 1000 days).

For each tau:
  - integrate the frozen-background wave (h_ref=30 m) + one-way second-order
    mean response u2 (per hb^2) to 1000 days;
  - build G0(z) and G2(z,t);
  - for each window, find hb_c^pred(tau) via B_pred>=0.
  - record secular-growth diagnostics of u2 / D_pred.

Also stores, for representative tau (10, 20, 50), the full u2_time / G2 so the
diagnostic-figure script can plot u0+hb^2 u2, Ucrit, Ucrit-u, min-over-layer.

Output:
  output/data/hm_model/second_order_boundary.npz
"""
import os, sys, time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.second_order_boundary as so

DATA = os.path.join(ROOT, "output", "data", "hm_model")

TAU_VALUES = np.arange(1, 62, 1, dtype=float)   # 61 values (matches sweep)
HB_REF = 30.0
N_DAYS = 1000
WINDOWS = so.WINDOWS_DAYS
REP_TAUS = [10.0, 20.0, 50.0]
N_WORKERS = 32


def _one_tau(tau):
    psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(
        HB_REF, tau, n_days=N_DAYS)
    G0, Ucrit0, betae0 = so.G0_profile()
    G2 = so.G2_field(u2_time)

    # predicted hbc per window
    hbc = {}
    for w in WINDOWS:
        hbc[w], _, _ = so.predicted_hbc(G0, G2, w)

    # secular-growth diagnostics: layer-min of D_pred at a fixed reference hb,
    # and the raw layer-averaged |u2| over time
    zmask = so._layer_mask()
    # amplitude of u2 in the layer as a function of time (per hb^2)
    u2_layer_amp = np.max(np.abs(u2_time[zmask, :]), axis=0)   # (nt,)
    # B_pred(hb) curve at a fixed hb for the 4 windows (to show window drift)
    return dict(tau=tau, hbc=hbc, u2_layer_amp=u2_layer_amp,
                u2_time=(u2_time if tau in REP_TAUS else None),
                G2=(G2 if tau in REP_TAUS else None),
                G0=G0)


def main():
    t0 = time.time()
    print(f"Second-order boundary sweep: {len(TAU_VALUES)} tau x {N_DAYS} d, "
          f"windows={WINDOWS}, h_ref={HB_REF} m")
    with mp.Pool(min(N_WORKERS, len(TAU_VALUES))) as pool:
        results = pool.map(_one_tau, TAU_VALUES)
    results.sort(key=lambda r: r["tau"])

    ntau = len(TAU_VALUES)
    hbc_by_window = {w: np.full(ntau, np.nan) for w in WINDOWS}
    u2_layer_amp = np.zeros((ntau, N_DAYS))
    for idx, r in enumerate(results):
        for w in WINDOWS:
            hbc_by_window[w][idx] = r["hbc"][w]
        amp = r["u2_layer_amp"]
        u2_layer_amp[idx, :len(amp)] = amp

    G0 = results[0]["G0"]

    # representative fields
    rep = {}
    for r in results:
        if r["tau"] in REP_TAUS and r["u2_time"] is not None:
            rep[f"u2_time_tau{int(r['tau'])}"] = r["u2_time"]
            rep[f"G2_tau{int(r['tau'])}"] = r["G2"]

    save = dict(
        tau_values=TAU_VALUES,
        hb_ref=HB_REF, n_days=N_DAYS, windows=np.array(WINDOWS),
        G0=G0, Z=so.Z, U_BG=so.U_BG,
        barrier_z1_km=so.BARRIER_Z1_KM, barrier_z2_km=so.BARRIER_Z2_KM,
        KSTAR2=so.KSTAR2, K2=so.K2, EPS=so.EPS, F2N2=so.F2N2, H0=so.H0,
        u2_layer_amp=u2_layer_amp,
    )
    for w in WINDOWS:
        save[f"hbc_pred_w{w}"] = hbc_by_window[w]
    save.update(rep)

    out = os.path.join(DATA, "second_order_boundary.npz")
    np.savez(out, **save)
    print(f"\nSaved: {out}")
    for w in WINDOWS:
        arr = hbc_by_window[w]
        finite = np.isfinite(arr)
        if finite.any():
            print(f"  window {w:4d} d: hbc_pred finite for {finite.sum()}/{ntau} tau; "
                  f"range {np.nanmin(arr):.1f}..{np.nanmax(arr):.1f} m")
        else:
            print(f"  window {w:4d} d: hbc_pred never finite (barrier never opens)")
    print(f"Elapsed {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
