"""
scripts/_verify_dcrit_equivalence.py
====================================
Task 1: verify equivalence of the critical-wind margin D_crit(t) with the
existing propagation-barrier metric M(t) = min_{25-40km} m2(z,t; c=0).

NO figures written here; pure numeric verification / diagnosis.

Definitions
-----------
Code m2 (matches models.hm76.compute_m2, c=0):
    m2 = (N^2/f0^2) [ betae/(eps*u) - K^2 ],  K^2 = k^2 + l^2 + lR^-2

Manuscript critical wind (K_*^2 = K^2 - l^2):
    U_crit = [ beta - eps (f0^2/N^2)(u_zz - u_z/H) ] / (eps K_*^2)
    D_crit = min_{25-40km} (U_crit - u)

Also compute the K^2-consistent margin (exact sign-partner of code m2):
    U_crit_full = betae / (eps K^2)   [since sign(m2)=sign(betae/(eps K^2)-u) for eps*u>0]
    D_full = min_{25-40km} (U_crit_full - u)
"""
import os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "output", "data", "hm_model")

H0 = 7000.0
ENSQ = 4.0e-4  # N^2

Z1_KM, Z2_KM = 25.0, 40.0


def first_zero_crossing(t, y):
    """First time y goes from <=0 to >0 (linear interp). Returns None if never."""
    y = np.asarray(y, float)
    for i in range(1, len(y)):
        if np.isnan(y[i-1]) or np.isnan(y[i]):
            continue
        if y[i-1] <= 0.0 < y[i]:
            # linear interpolation
            frac = (0.0 - y[i-1]) / (y[i] - y[i-1])
            return t[i-1] + frac * (t[i] - t[i-1])
    return None


def analyze(fname, label):
    d = np.load(os.path.join(DATA, fname))
    z = d["z"]; z_km = z / 1000.0
    dz = float(d["dz"]); eps = float(d["eps"])
    k = float(d["k"]); l = float(d["l"]); f0 = float(d["f0"]); beta = float(d["beta"])
    betae = d["betae_time"]        # = beta - eps*qy  (nz, nt)
    ud = d["ud"]                   # zonal wind (nz, nt)
    nt = ud.shape[1]

    lRm2 = f0**2 / (4.0 * ENSQ * H0**2)
    K2 = k**2 + l**2 + lRm2
    Ks2 = K2 - l**2

    mask = (z_km >= Z1_KM) & (z_km <= Z2_KM)
    idx = np.where(mask)[0]
    z_layer_km = z_km[idx]

    t = np.arange(nt, dtype=float)

    M = np.full(nt, np.nan)          # code m2 min
    Dcrit = np.full(nt, np.nan)      # manuscript K_*^2 margin
    Dfull = np.full(nt, np.nan)      # K^2-consistent margin
    epsu_pos_all = np.zeros(nt, dtype=bool)
    z_bottleneck = np.full(nt, np.nan)   # height (km) of argmin(U_crit-u)

    for j in range(nt):
        u = ud[:, j]
        be = betae[:, j]
        # code m2 over layer
        denom = eps * u[idx]
        m2 = (ENSQ / f0**2) * (be[idx] / denom - K2)
        m2 = np.where(np.abs(denom) < 1e-6, np.nan, m2)
        M[j] = np.nanmin(m2)

        # u derivatives (centered interior); needed at layer points
        u_z = np.gradient(u, dz)
        u_zz = np.gradient(u_z, dz)
        # U_crit (manuscript, K_*^2). Numerator uses beta (not betae): the
        # -eps(f0^2/N^2)(u_zz - u_z/H) term IS the curvature part of eps*qy,
        # and qy = -l^2 u + (f0^2/N^2)(u_zz - u_z/H). So beta - eps*(curv) =
        # betae + eps*l^2*u. Keep it EXACTLY as the manuscript writes it.
        curv = (f0**2 / ENSQ) * (u_zz - u_z / H0)
        Ucrit = (beta - eps * curv) / (eps * Ks2)
        marg = (Ucrit - u)[idx]
        Dcrit[j] = np.min(marg)
        z_bottleneck[j] = z_layer_km[int(np.argmin(marg))]

        # K^2-consistent margin: Ucrit_full = betae/(eps K^2)
        Ucrit_full = be / (eps * K2)
        Dfull[j] = np.min((Ucrit_full - u)[idx])

        epsu_pos_all[j] = np.all(eps * u[idx] > 0)

    tM = first_zero_crossing(t, M)
    tD = first_zero_crossing(t, Dcrit)
    tF = first_zero_crossing(t, Dfull)

    print(f"\n===== {label}  ({fname}) =====")
    print(f"  K^2={K2:.4e}  K_*^2={Ks2:.4e}  l^2={l**2:.4e}  eps={eps}")
    print(f"  first zero crossing  M(t)      [code m2] : {tM}")
    print(f"  first zero crossing  D_crit(t) [K_*^2]   : {tD}")
    print(f"  first zero crossing  D_full(t) [K^2]     : {tF}")
    if tM is not None and tD is not None:
        print(f"  |t_open(D_crit) - t_open(M)| = {abs(tD - tM):.4f} d")
    if tM is not None and tF is not None:
        print(f"  |t_open(D_full)  - t_open(M)| = {abs(tF - tM):.4f} d")

    # eps*u>0 check at/before t_open
    if tM is not None:
        jopen = int(np.ceil(tM))
        frac_ok = np.mean(epsu_pos_all[:jopen+1])
        print(f"  eps*u>0 in layer for ALL days up to day {jopen}: "
              f"{np.all(epsu_pos_all[:jopen+1])} (frac={frac_ok:.3f})")
        # bottleneck height near opening
        win = slice(max(0, jopen-8), jopen+1)
        zb = z_bottleneck[win]
        print(f"  bottleneck z (km) days {win.start}-{jopen}: "
              f"min={np.nanmin(zb):.1f} max={np.nanmax(zb):.1f} "
              f"at_open={z_bottleneck[jopen]:.1f}")
    else:
        print("  M(t) never crosses zero -> barrier never opens (no-transition).")
        print(f"  D_crit crosses zero: {tD is not None}; D_full crosses: {tF is not None}")
        print(f"  max M={np.nanmax(M):.3e}  max D_crit={np.nanmax(Dcrit):.3f} "
              f"max D_full={np.nanmax(Dfull):.3f}")

    return dict(t=t, M=M, Dcrit=Dcrit, Dfull=Dfull, epsu_pos=epsu_pos_all,
                z_bottleneck=z_bottleneck, tM=tM, tD=tD, tF=tF)


if __name__ == "__main__":
    r10 = analyze("sweep_hb30_tau10.00.npz", "TRANSITION  hb=30 tau=10")
    r20 = analyze("sweep_hb30_tau20.00.npz", "NO-TRANSITION hb=30 tau=20")

    print("\n===== manuscript reference =====")
    print("  t_A=29.2 d  t_open=31.4 d  t_rev=50.1 d (native 29.25/31.375/50.0625)")
