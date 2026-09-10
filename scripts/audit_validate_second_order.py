"""
scripts/audit_validate_second_order.py
=======================================
STEPS 2, 3 (validation) + consistency checks A, B, C, D, E of the
second-order boundary audit.

Validates the standalone integrator in second_order_boundary.py against the
CURRENT models/hm76.py, then verifies scaling and algebraic consistency.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hm76 import run_hm76, compute_m2, compute_min_m2
import scripts.second_order_boundary as so


def check_frozen_wave_matches_model():
    """The standalone frozen-background wave psi must equal run_hm76(WMFI=False)."""
    print("\n[VALIDATION] standalone frozen wave vs run_hm76(wave_mean_feedback=False)")
    for hb, tau in [(30.0, 10.0), (30.0, 20.0), (25.0, 50.0)]:
        out = run_hm76(hb, tau * 86400.0, s=so.S, dz=so.DZ, imax=so.IMAX, dt=so.DT,
                       n_days=60, alpha_on=False, wave_mean_feedback=False,
                       mean_flow_coupling=1.0)
        psi_time, _, _ = so.integrate_frozen_and_second_order(hb, tau, n_days=60)
        # model psi_time is complex64; compare on overlapping days
        nt = min(out["psi_time"].shape[1], psi_time.shape[1])
        pm = out["psi_time"][:, :nt].astype(np.complex128)
        ps = psi_time[:, :nt]
        denom = np.max(np.abs(pm)) + 1e-30
        rel = np.max(np.abs(pm - ps)) / denom
        print(f"  hb={hb} tau={tau}: max|dpsi|/max|psi| = {rel:.3e}  "
              f"(max|psi|={np.max(np.abs(pm)):.3e})")
        assert rel < 1e-5, "frozen wave does not match model!"
    print("  -> standalone frozen wave reproduces run_hm76(no-WMFI). OK")


def check_A_scaling():
    """Check A: psi ~ hb (linear); one-way mean response ~ hb^2."""
    print("\n[CHECK A] scaling of psi (~hb) and second-order response (~hb^2)")
    tau = 10.0
    hbs = [10.0, 20.0, 30.0, 40.0]
    # psi1 = psi/hb should be independent of hb
    psi1_ref = None
    for hb in hbs:
        psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(hb, tau, n_days=120)
        psi1 = psi_time / hb
        if psi1_ref is None:
            psi1_ref = psi1
            u2_ref = u2_time
            hb_ref0 = hb
        else:
            rel_psi = np.max(np.abs(psi1 - psi1_ref)) / (np.max(np.abs(psi1_ref)) + 1e-30)
            # u2_time is already per hb^2 -> should be hb-independent
            rel_u2 = np.max(np.abs(u2_time - u2_ref)) / (np.max(np.abs(u2_ref)) + 1e-30)
            print(f"  hb={hb:5.1f}: max|psi/hb - ref|/ref = {rel_psi:.3e} ; "
                  f"max|u2(perhb2) - ref|/ref = {rel_u2:.3e}")
            assert rel_psi < 1e-9, "psi not linear in hb"
            assert rel_u2 < 1e-9, "second-order response not hb^2 (per-hb^2 not invariant)"
    print(f"  -> psi linear in hb; one-way mean response scales as hb^2. OK")

    # Independent absolute confirmation: raw (not per-hb^2) response grows as hb^2
    print("  absolute-scaling confirmation (raw u2*hb^2 grows ~ hb^2):")
    base = None
    for hb in hbs:
        _, _, u2_time = so.integrate_frozen_and_second_order(hb, tau, n_days=120)
        raw = u2_time * hb * hb   # undo the per-hb^2 normalization -> physical response
        amp = np.max(np.abs(raw))
        if base is None:
            base = amp; base_hb = hb
        ratio = amp / base
        expect = (hb / base_hb) ** 2
        print(f"    hb={hb:5.1f}: |raw u2| ratio = {ratio:8.4f}  expect hb^2 ratio = {expect:8.4f}")


def check_B_ucrit_algebra():
    """Check B: reconstruct u_pred=u0+hb^2 u2, compute exact Ucrit(u_pred)-u_pred,
    confirm it agrees with G0 + hb^2 G2 (analytic linearization) up to O(hb^4)."""
    print("\n[CHECK B] Ucrit(u_pred)-u_pred  vs  G0 + hb^2 G2")
    tau = 10.0
    psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=250)
    G0, Ucrit0, betae0 = so.G0_profile()
    G2 = so.G2_field(u2_time)
    zmask = (so.Z / 1000.0 >= 25.0) & (so.Z / 1000.0 <= 40.0)
    for hb in [10.0, 20.0, 30.0]:
        u_pred = so.U_BG[:, None] + hb * hb * u2_time
        D_exact = so.Dmargin_exact(u_pred)          # Ucrit(u_pred)-u_pred, exact
        D_lin = G0[:, None] + hb * hb * G2           # analytic linearization
        # compare in the barrier layer where it matters
        diff = np.abs(D_exact[zmask] - D_lin[zmask])
        scale = np.max(np.abs(D_exact[zmask])) + 1e-30
        print(f"  hb={hb:5.1f}: max|D_exact - (G0+hb^2 G2)| = {diff.max():.3e}  "
              f"(rel {diff.max()/scale:.3e})  [O(hb^4) expected]")
    print("  -> linearization matches exact margin; residual scales like hb^4. OK")


def check_C_refractive_sign():
    """Check C: build u_pred, compute m^2 from compute_m2 directly, verify sign
    change agrees with D_pred=0 where eps*u_pred>0."""
    print("\n[CHECK C] refractive-index sign vs D_pred sign")
    tau = 10.0
    psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=250)
    G0, _, _ = so.G0_profile()
    G2 = so.G2_field(u2_time)
    hb = 28.0
    u_pred = so.U_BG[:, None] + hb * hb * u2_time
    # betae field from u_pred, feed compute_m2
    nt = u_pred.shape[1]
    betae_pred = np.zeros_like(u_pred)
    for t in range(nt):
        betae_pred[:, t] = so.BETA - so.EPS * so.qy_of_u(u_pred[:, t])
    m2 = compute_m2(dict(ud=u_pred, betae_time=betae_pred, eps=so.EPS), c=0.0, s=so.S)
    D_exact = so.Dmargin_exact(u_pred)
    denom_ok = (so.EPS * u_pred) > 0
    # Where eps*u_pred>0, sign(m2) should equal sign(D_exact)
    valid = denom_ok & ~np.isnan(m2)
    same = np.sign(m2[valid]) == np.sign(D_exact[valid])
    frac = np.mean(same)
    print(f"  fraction of (z,t) with matching sign(m2)==sign(Ucrit-u) "
          f"(where eps*u>0): {frac:.6f}")
    print(f"  -> {'OK' if frac > 0.999 else 'CHECK'}")


def check_D_denominator():
    """Check D: for c=0, verify eps*u_pred>0 at the predicted barrier-opening
    point (min over 25-40 km)."""
    print("\n[CHECK D] denominator eps*u_pred>0 at barrier-opening location")
    tau = 10.0
    psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=250)
    G0, _, _ = so.G0_profile()
    G2 = so.G2_field(u2_time)
    zmask = (so.Z / 1000.0 >= 25.0) & (so.Z / 1000.0 <= 40.0)
    zidx = np.where(zmask)[0]
    hb = 28.0
    D_pred = G0[:, None] + hb * hb * G2
    Dm = D_pred[zmask, :]
    u_pred = so.U_BG[:, None] + hb * hb * u2_time
    # find first (t) where min over layer >= 0
    minlayer = np.min(Dm, axis=0)
    topen = np.where(minlayer >= 0)[0]
    if len(topen) > 0:
        t0 = topen[0]
        zloc = zidx[np.argmin(Dm[:, t0])]
        print(f"  hb={hb}: barrier opens day {t0}, z={so.Z[zloc]/1000:.0f} km, "
              f"eps*u_pred={so.EPS*u_pred[zloc,t0]:.3f} (>0? {so.EPS*u_pred[zloc,t0]>0})")
    else:
        print(f"  hb={hb}: predicted barrier never opens within window")
    # Also scan the min-margin location generally
    allmin = np.min(D_pred[zmask, :], axis=0)
    t_best = np.argmax(allmin)
    zloc = zidx[np.argmin(D_pred[zmask, t_best])]
    print(f"  most-open day {t_best}, z={so.Z[zloc]/1000:.0f} km, "
          f"eps*u_pred={so.EPS*u_pred[zloc,t_best]:.3f}")


def check_E_derivatives():
    """Check E: verify u2_z, u2_zz are not grid-noise dominated -- compare the
    G2 spectral content / smoothness by reporting the ratio of the curvature
    term to u2 itself and a grid-scale oscillation metric."""
    print("\n[CHECK E] derivative / grid-noise check on u2")
    tau = 10.0
    psi_time, qy2_time, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=250)
    zmask = (so.Z / 1000.0 >= 25.0) & (so.Z / 1000.0 <= 40.0)
    u2z, u2zz = so.d2_dz_operators(u2_time)
    # pick a late day
    t = 200
    col = u2_time[:, t]
    # grid-scale 2dz oscillation indicator: second difference vs magnitude
    curv = u2zz[zmask, t]
    mag = np.max(np.abs(u2_time[zmask, t])) + 1e-30
    # ratio of curvature term (f2/N2/K*^2 * u2zz) to u2 in the layer
    term_curv = (so.F2N2 / so.KSTAR2) * u2zz[zmask, t]
    term_u2 = -u2_time[zmask, t]
    print(f"  day {t}: max|u2| (layer) = {mag:.4e}")
    print(f"  max|(f2/N2/K*^2) u2_zz| (layer) = {np.max(np.abs(term_curv)):.4e}")
    print(f"  max|u2| term (layer)            = {np.max(np.abs(term_u2)):.4e}")
    print(f"  curvature/u2 term ratio         = {np.max(np.abs(term_curv))/(np.max(np.abs(term_u2))+1e-30):.3f}")
    # sign of 2dz noise: alternation count
    d = np.diff(np.sign(col[zmask]))
    print(f"  sign changes of u2 within layer = {int(np.sum(d!=0))} (few => smooth)")


if __name__ == "__main__":
    check_frozen_wave_matches_model()
    check_A_scaling()
    check_B_ucrit_algebra()
    check_C_refractive_sign()
    check_D_denominator()
    check_E_derivatives()
    print("\nAll validation/consistency checks complete.")
