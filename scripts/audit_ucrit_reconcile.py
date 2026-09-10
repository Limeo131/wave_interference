"""
scripts/audit_ucrit_reconcile.py
================================
Focused re-audit (ISSUE 1): reconcile the U_crit algebra and recompute the
second-order predicted boundary using ONLY the manuscript definition.

Manuscript definition (stationary wave, c=0), from
    m^2 = (N^2/f0^2)[ beta_e/(eps u) - K^2 ],
    beta_e = beta + eps l^2 u - eps (f0^2/N^2)(u_zz - u_z/H),
rearranged exactly (eps u > 0) as
    m^2 = (N^2/f0^2) [ eps K*^2/(eps u) ] (U_crit - u),
    K*^2 = K^2 - l^2,
    U_crit(u) = [ beta - eps (f0^2/N^2)(u_zz - u_z/H) ] / (eps K*^2).
=> m^2 > 0  <=>  U_crit - u > 0.

KEY: this U_crit is LINEAR in u and its z-derivatives, so
    U_crit(u0 + h^2 u2) - (u0 + h^2 u2)  ==  G0 + h^2 G2   exactly,
    G0 = U_crit(u0) - u0,
    G2 = -u2 - (f0^2/(N^2 K*^2)) (u2_zz - u2_z/H).
There is NO intrinsic O(h^4) residual (unlike the previous K^2-based D_code).

This script:
  1. verifies the algebra numerically (m^2 from compute_m2 vs the rearranged
     eps K*^2/(eps u) (U_crit - u) form);
  2. proves the exact-vs-expanded identity D_pred = G0 + h^2 G2 to machine
     precision for tau in {10,20,50}, hb in {10,20,30,40};
  3. recomputes h_{b,c}^pred(tau) (1000-day window) with the manuscript D_pred;
  4. compares old vs corrected predicted boundary + error metrics.

Reuses the VALIDATED frozen-wave + one-way u2 integrator from
second_order_boundary.py (that part is unchanged and correct); only the
U_crit / margin definitions are corrected here.

Outputs:
  output/data/hm_model/second_order_boundary_corrected.npz
  output/data/hm_model/second_order_boundary_comparison_corrected.csv
"""
import os, sys, csv
import numpy as np
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.second_order_boundary as so
from models.hm76 import compute_m2

DATA = os.path.join(ROOT, "output", "data", "hm_model")

# constants (mirror so.*)
EPS, F2N2, H0, DZ = so.EPS, so.F2N2, so.H0, so.DZ
K2, KSTAR2, L = so.K2, so.KSTAR2, so.L
BETA = so.BETA
Z = so.Z
U_BG = so.U_BG


# ── manuscript U_crit (linear in u), c = 0 ───────────────────────────────────
def ucrit_manuscript(u_field):
    """U_crit(u) = [beta - eps (f0^2/N^2)(u_zz - u_z/H)] / (eps K*^2).
    Accepts (nz,) or (nz,nt). Uses centered interior / one-sided edge z-derivs
    consistent with the model dz (same operator as G2)."""
    single = (u_field.ndim == 1)
    U = u_field[:, None] if single else u_field
    uz, uzz = so.d2_dz_operators(U)
    Uc = (BETA - EPS * F2N2 * (uzz - uz / H0)) / (EPS * KSTAR2)
    return Uc[:, 0] if single else Uc


def D_pred_exact(u_field):
    """Manuscript margin U_crit(u) - u (exact, for any u field)."""
    return ucrit_manuscript(u_field) - u_field


def G0_manuscript():
    return ucrit_manuscript(U_BG) - U_BG


def G2_manuscript(u2_time):
    u2z, u2zz = so.d2_dz_operators(u2_time)
    return -u2_time - (F2N2 / KSTAR2) * (u2zz - u2z / H0)


# =============================================================================
# 1. Algebra verification: rearranged m^2 identity
# =============================================================================
def verify_algebra():
    print("=" * 70)
    print("1. U_crit ALGEBRA VERIFICATION (rearranged m^2 identity)")
    print("=" * 70)
    # Build an arbitrary physical wind field (day-0 bg + a smooth perturbation)
    # and check m^2 (compute_m2) == (N^2/f0^2)(eps K*^2/(eps u))(U_crit - u).
    _, _, u2_time = so.integrate_frozen_and_second_order(30.0, 10.0, n_days=60)
    u = U_BG[:, None] + 400.0 * u2_time  # exaggerate to test generic u
    nz, nt = u.shape
    betae = np.zeros_like(u)
    for t in range(nt):
        betae[:, t] = BETA - EPS * so.qy_of_u(u[:, t])
    m2_code = compute_m2(dict(ud=u, betae_time=betae, eps=EPS), c=0.0, s=so.S)
    N2_f2 = so.ENSQ / so.F0 ** 2
    Uc = ucrit_manuscript(u)
    m2_rearr = N2_f2 * (EPS * KSTAR2 / (EPS * u)) * (Uc - u)
    # compare on interior where compute_m2 is defined and eps*u>0
    valid = (~np.isnan(m2_code)) & (EPS * u > 0)
    # exclude the two z-edges (compute_m2 uses betae which needs qy interior)
    valid[0, :] = valid[-1, :] = False
    diff = np.abs(m2_code[valid] - m2_rearr[valid])
    scale = np.max(np.abs(m2_code[valid]))
    print(f"  max|m2_code - m2_rearranged| = {diff.max():.3e}  (scale {scale:.3e}, "
          f"rel {diff.max()/scale:.3e})")
    print(f"  -> algebra {'VERIFIED at numerical precision' if diff.max()/scale < 1e-6 else 'MISMATCH'}")
    return diff.max() / scale


# =============================================================================
# 2. Exact-vs-expanded identity  D_pred = G0 + h^2 G2  (must be machine precision)
# =============================================================================
def identity_test():
    print("\n" + "=" * 70)
    print("2. IDENTITY TEST: U_crit(u0+h^2 u2)-(u0+h^2 u2)  vs  G0 + h^2 G2")
    print("=" * 70)
    zmask = so._layer_mask()
    rows = []
    worst = 0.0
    for tau in [10.0, 20.0, 50.0]:
        _, _, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=250)
        G0 = G0_manuscript()
        G2 = G2_manuscript(u2_time)
        for hb in [10.0, 20.0, 30.0, 40.0]:
            u_pred = U_BG[:, None] + hb * hb * u2_time
            D_exact = D_pred_exact(u_pred)
            D_lin = G0[:, None] + hb * hb * G2
            d = np.abs(D_exact - D_lin)
            # report both full-domain and in-layer
            dmax = float(d.max())
            dlayer = float(d[zmask].max())
            scale = float(np.max(np.abs(D_exact)) + 1e-30)
            rel = dmax / scale
            worst = max(worst, rel)
            rows.append((tau, hb, dmax, dlayer, rel))
            print(f"  tau={tau:4.0f} hb={hb:4.0f}: max|diff|={dmax:.3e}  "
                  f"in-layer={dlayer:.3e}  rel={rel:.3e}")
    print(f"\n  worst relative difference across all cases: {worst:.3e}")
    ok = worst < 1e-10
    print(f"  -> identity holds at {'MACHINE PRECISION' if ok else 'NON-TRIVIAL RESIDUAL (STOP+DIAGNOSE)'}")
    return ok, rows


# =============================================================================
# 3. Corrected predicted boundary (manuscript D_pred), 1000-day window
# =============================================================================
def corrected_boundary():
    print("\n" + "=" * 70)
    print("3. CORRECTED PREDICTED BOUNDARY (manuscript U_crit, 1000-day window)")
    print("=" * 70)
    TAU = np.arange(1, 62, 1, dtype=float)
    G0 = G0_manuscript()
    zmask = so._layer_mask()

    def B_pred_of_hb(G2, hb):
        D = G0[zmask, None] + hb * hb * G2[zmask, :]
        return float(np.max(np.min(D, axis=0)))

    def hbc(G2, hb_lo=1.0, hb_hi=400.0):
        grid = np.arange(hb_lo, hb_hi + 0.5, 0.5)
        B = np.array([B_pred_of_hb(G2, h) for h in grid])
        pos = np.where(B >= 0)[0]
        if len(pos) == 0:
            return np.nan
        first = pos[0]
        if first == 0:
            return float(grid[0])
        lo, hi = grid[first - 1], grid[first]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if B_pred_of_hb(G2, mid) >= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-3:
                break
        return float(0.5 * (lo + hi))

    hbc_corr = np.full(len(TAU), np.nan)
    for i, tau in enumerate(TAU):
        _, _, u2_time = so.integrate_frozen_and_second_order(30.0, tau, n_days=1000)
        G2 = G2_manuscript(u2_time)
        hbc_corr[i] = hbc(G2)
    print(f"  corrected hbc_pred range: {np.nanmin(hbc_corr):.1f}..{np.nanmax(hbc_corr):.1f} m "
          f"(finite {np.isfinite(hbc_corr).sum()}/{len(TAU)})")
    return TAU, hbc_corr, G0


if __name__ == "__main__":
    rel_alg = verify_algebra()
    ok_id, id_rows = identity_test()
    if not ok_id:
        print("\nIDENTITY TEST FAILED -- stopping before boundary recompute.")
        sys.exit(1)
    TAU, hbc_corr, G0 = corrected_boundary()

    # reference + old prediction
    ref = np.load(os.path.join(DATA, "audit_reference_boundaries.npz"))
    old = np.load(os.path.join(DATA, "second_order_boundary.npz"), allow_pickle=True)
    btau = ref["boundary_transition_tau"]; bhb = ref["boundary_transition_hb"]
    mtau = ref["boundary_Mmax0_tau"]; mhb = ref["boundary_Mmax0_hb"]
    act = np.array([bhb[np.where(np.isclose(btau, t))[0][0]] if np.any(np.isclose(btau, t)) else np.nan for t in TAU])
    mm0 = np.array([mhb[np.where(np.isclose(mtau, t))[0][0]] if np.any(np.isclose(mtau, t)) else np.nan for t in TAU])
    old_hbc = old["hbc_pred_w1000"]

    # metrics vs actual
    e = hbc_corr - act
    fin = np.isfinite(e)
    mae = np.mean(np.abs(e[fin])); rms = np.sqrt(np.mean(e[fin] ** 2))
    bias = np.mean(e[fin]); r = pearsonr(act[fin], hbc_corr[fin])[0]
    within1 = np.mean(np.abs(e[fin]) <= 1); within2 = np.mean(np.abs(e[fin]) <= 2)
    within5 = np.mean(np.abs(e[fin]) <= 5)

    print("\n" + "=" * 70)
    print("4. OLD vs CORRECTED vs ACTUAL (selected tau)")
    print("=" * 70)
    print(f"{'tau':>4} {'actual':>7} {'Mmax0':>7} {'old_pred':>9} {'corr_pred':>10} {'corr-act':>9}")
    for t in [1, 10, 20, 30, 50, 61]:
        i = int(t - 1)
        print(f"{t:4d} {act[i]:7.2f} {mm0[i]:7.2f} {old_hbc[i]:9.2f} {hbc_corr[i]:10.2f} {hbc_corr[i]-act[i]:9.2f}")
    print(f"\n  vs ACTUAL: bias={bias:+.2f} m  MAE={mae:.2f} m  RMS={rms:.2f} m  "
          f"Pearson r={r:.4f}")
    print(f"  within 1m={within1:.2f}  2m={within2:.2f}  5m={within5:.2f}")

    # save
    np.savez(os.path.join(DATA, "second_order_boundary_corrected.npz"),
             tau_values=TAU, hbc_pred_corrected_w1000=hbc_corr,
             hbc_pred_old_w1000=old_hbc, actual=act, Mmax0=mm0, G0=G0,
             algebra_rel_diff=rel_alg,
             bias=bias, mae=mae, rms=rms, pearson=r,
             within1=within1, within2=within2, within5=within5)
    csvp = os.path.join(DATA, "second_order_boundary_comparison_corrected.csv")
    with open(csvp, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tau_days", "hbc_actual_m", "hbc_Mmax0_m",
                     "hbc_pred_old_w1000_m", "hbc_pred_corrected_w1000_m",
                     "corrected_minus_actual_m"])
        for i, t in enumerate(TAU):
            wr.writerow([f"{t:.0f}", f"{act[i]:.3f}", f"{mm0[i]:.3f}",
                         f"{old_hbc[i]:.3f}", f"{hbc_corr[i]:.3f}",
                         f"{hbc_corr[i]-act[i]:.3f}"])
    print(f"\n  saved {os.path.join(DATA,'second_order_boundary_corrected.npz')}")
    print(f"  saved {csvp}")
