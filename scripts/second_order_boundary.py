"""
scripts/second_order_boundary.py
================================
STEPS 2-6 of the second-order (weakly nonlinear) transition-boundary
feasibility audit.

Goal
----
Test whether a ONE-WAY second-order calculation predicts the (hb, tau)
transition boundary WITHOUT using the full nonlinear transition result.

Method (all consistent with the CURRENT models/hm76.py discretization)
----------------------------------------------------------------------
1. Frozen-background first-order wave:
     Integrate the wave PV q' and streamfunction psi' on the FROZEN day-0
     background u0(z) (the no-WMFI configuration), reproducing exactly the
     hm76.py wave step (AB3 on q', tridiagonal psi-inversion, lower BC
     psi(0,t)=hb (g/f0)(1-exp(-t/tau)), radiation-type top).
     The frozen-background wave is linear in hb, so we define
        psi1 = psi_ref / h_ref        (per unit hb)
     with h_ref = 30 m (linearity verified explicitly, check A).

2. One-way second-order mean-flow response u2 (per unit hb^2):
     Using psi1, compute the SAME wave->mean forcing term the model uses,
        F(z,t) = 0.5 l^2 k eps (f0^2/N^2) e^{z/H}
                 Im( psi1_i conj(psi1_{i+1}+psi1_{i-1}-2 psi1_i)/dz^2 ),
     integrate the mean PV-gradient anomaly qy2 in time with the SAME AB3
     scheme and dt, then invert the SAME diagnostic PV relation
        qy = -l^2 U + (f0^2/N^2)(U_zz - U_z/H)
     with the SAME tridiagonal solver, but with HOMOGENEOUS anomaly BCs:
        u2(z_b)=0 (lower), Neumann top (matches the model's mean upper BC).
     Because F is quadratic in psi1, qy2 and u2 are the response PER UNIT
     hb^2.  Crucially, u2 is NEVER fed back into the wave step (one-way).

3. Predicted propagation margin (stationary wave, c=0), consistent with the
   code's m^2 = (N^2/f0^2)[ betae/(eps U) - K^2 ]:
     m^2 = 0  <=>  U = Ucrit(U) := betae(U)/(eps K^2),
     where betae(U) = beta - eps qy(U) and qy(U) = -l^2 U + (f2/N2)(U_zz-U_z/H).
     Define the exact code-consistent margin
        Dmargin(U) := Ucrit(U) - U.
     Expanding U = u0 + hb^2 u2 to O(hb^2):
        Dmargin ≈ G0(z) + hb^2 G2(z,t;tau),
        G0 = Ucrit(u0) - u0,
        G2 = -u2 - (f2/(N2 K*^2))[u2_zz - u2_z/H],   K*^2 = k^2 + 1/LD^2
             (the linearization of Ucrit(U)-U about u0; derivation in report).
   We use BOTH:
     - the analytic linearization D_pred = G0 + hb^2 G2   (the requested form)
     - the exact reconstruction Dmargin(u0 + hb^2 u2)      (consistency check B)

4. Predicted barrier metric:
     B_pred(hb,tau) = max_{t in window} [ min_{25<=z<=40 km} D_pred(z,t) ].
     Barrier can open when B_pred >= 0.  Smallest hb with B_pred>=0 is
     hb_c^pred(tau), found by interpolation in hb (B_pred is a quadratic in
     hb^2 at fixed (z,t), so hb_c^2 = -min_t..  solved per the reduction below).

Windows tested: 150, 250, 500, 1000 days.

Outputs
-------
  output/data/hm_model/second_order_boundary.npz   (all arrays)
  Console log of consistency checks A-E.

This script does NOT modify models/hm76.py or any Fig-6 script/data.
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hm76 import run_hm76, compute_m2, compute_min_m2

DATA = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA, exist_ok=True)

# ── Model constants (identical to models/hm76.py) ───────────────────────────
S = 2.0
DZ = 1000.0
IMAX = 71
DT = 360.0 * 15.0            # 5400 s
A_EARTH = 6378000.0
K = S / (A_EARTH * np.cos(np.pi / 3.0))
L = 3.0 / A_EARTH
OMEGA_E = 7.29e-5
G = 9.81
F0 = 2.0 * OMEGA_E * np.sin(np.pi / 3.0)
BETA = 2.0 * OMEGA_E * np.cos(np.pi / 3.0) / A_EARTH
ENSQ = 4.0e-4                # N^2
H0 = 7000.0
EPS = 8.0 / (3.0 * np.pi)
LRMSQ = (F0 * F0) / (4.0 * ENSQ * H0 * H0)   # 1/LD^2
K2 = K * K + L * L + LRMSQ                    # full horizontal+deformation operator
KSTAR2 = K * K + LRMSQ                         # K^2 - l^2  (appears in G2)
R = (ENSQ * DZ * DZ) / (F0 * F0)
P = 2.0 + R * (K * K + L * L + LRMSQ)
PU = 2.0 + R * L * L
VP = 1.0 + 0.5 * DZ / H0
VM = 1.0 - 0.5 * DZ / H0
F2N2 = F0 * F0 / ENSQ
STEPS_PER_DAY = int(3600 * 24.0 / DT)         # 16

Z = DZ * np.arange(IMAX) + 10000.0
U_BG = 12.0 + (Z - 10000.0) * (52.0 - 12.0) / 40000.0   # frozen day-0 background


# ─────────────────────────────────────────────────────────────────────────────
def qy_of_u(u):
    """Discrete diagnostic PV-gradient qy = -l^2 U + (f2/N2)(U_zz - U_z/H).
    Interior only (matches hm76.py init lines 172-176). Endpoints = 0."""
    qy = np.zeros(IMAX)
    for i in range(1, IMAX - 1):
        qy[i] = (-L * L * u[i]
                 + F2N2 * ((u[i + 1] + u[i - 1] - 2.0 * u[i]) / (DZ * DZ)
                           - (u[i + 1] - u[i - 1]) / (2.0 * DZ * H0)))
    return qy


def invert_qy_to_u(qy, lower_bc):
    """Invert qy -> U with the model's tridiagonal solver (hm76.py lines 220-227).
    vp U_{i+1} - pu U_i + vm U_{i-1} = r qy_i, Neumann-type top, U(z_b)=lower_bc."""
    ck = np.zeros(IMAX); dk = np.zeros(IMAX)
    ck[IMAX - 2] = 1.0; dk[IMAX - 2] = 0.0
    for i in range(IMAX - 2, -1, -1):
        ck[i - 1] = -VP / (ck[i] * VM - PU)
        dk[i - 1] = (R * qy[i] - dk[i] * VM) / (ck[i] * VM - PU)
    u = np.zeros(IMAX)
    u[0] = lower_bc
    for i in range(IMAX - 1):
        u[i + 1] = ck[i] * u[i] + dk[i]
    return u


# ─────────────────────────────────────────────────────────────────────────────
def integrate_frozen_and_second_order(hb_ref, tau_days, n_days,
                                       return_full=False):
    """
    Integrate, on the frozen day-0 background:
      (i)  the linear wave psi'(z,t) forced by hb_ref (frozen-background,
           reproducing hm76.py wave step exactly);
      (ii) the ONE-WAY second-order mean PV-gradient anomaly qy2 and wind u2
           PER UNIT hb^2, forced by psi1 = psi/hb_ref, integrated with the
           SAME AB3 scheme, never fed back into the wave.

    Returns daily-sampled arrays:
      psi_time   (imax, ndays)  complex  -- frozen-background wave at hb_ref
      u2_time    (imax, ndays)  real     -- second-order wind response / hb^2
      qy2_time   (imax, ndays)  real     -- second-order PV-grad anomaly / hb^2
    """
    tau = tau_days * 86400.0
    mmax = n_days * STEPS_PER_DAY
    ndays_out = int((mmax - 1) / STEPS_PER_DAY) + 1

    betae_bg = BETA - EPS * qy_of_u(U_BG)

    # wave buffers (4-level AB3)
    psi = np.zeros((IMAX, 4), dtype=np.complex128)
    q = np.zeros((IMAX, 4), dtype=np.complex128)
    # second-order mean buffers (per unit hb^2): forcing built from psi1=psi/hb_ref
    qy2 = np.zeros((IMAX, 4))            # PV-gradient anomaly / hb^2
    ak = np.zeros(IMAX); bk = np.zeros(IMAX, dtype=np.complex128)

    psi_time = np.zeros((IMAX, ndays_out), dtype=np.complex128)
    u2_time = np.zeros((IMAX, ndays_out))
    qy2_time = np.zeros((IMAX, ndays_out))

    inv_hb2 = 1.0 / (hb_ref * hb_ref)   # convert psi-quadratic flux -> per hb^2

    m = 0; mm = 0
    while m < mmax:
        t = DT * m
        # ---- wave PV tendency (frozen background), hm76.py lines 195-205 ----
        for i in range(1, IMAX - 1):
            be = betae_bg[i]
            uadv1 = -1j * K * EPS * U_BG[i] * q[i, 0]
            uadv2 = -1j * K * EPS * U_BG[i] * q[i, 1]
            uadv3 = -1j * K * EPS * U_BG[i] * q[i, 2]
            vadv1 = -be * 1j * K * psi[i, 0]
            vadv2 = -be * 1j * K * psi[i, 1]
            vadv3 = -be * 1j * K * psi[i, 2]
            q[i, 3] = q[i, 2] + (DT / 12.0) * (5.0 * uadv1 - 16.0 * uadv2 + 23.0 * uadv3
                                               + 5.0 * vadv1 - 16.0 * vadv2 + 23.0 * vadv3)
            # alpha_on=False -> no damping terms
        # ---- invert psi from q (hm76.py lines 214-219) ----
        ak[IMAX - 2] = 0.0; bk[IMAX - 2] = 0.0 + 0.0j
        for i in range(IMAX - 2, -1, -1):
            ak[i - 1] = -1.0 / (ak[i] - P)
            bk[i - 1] = (R * q[i, 3] - bk[i]) / (ak[i] - P)
        psi[0, 3] = hb_ref * (G / F0) * (1.0 - np.exp(-t / tau))
        for i in range(IMAX - 1):
            psi[i + 1, 3] = ak[i] * psi[i, 3] + bk[i]

        # ---- one-way second-order mean forcing (per hb^2), from psi1=psi/hb_ref ----
        # F uses psi at the SAME three past time levels as the model's AB3 flux.
        # We build the flux from psi (hb_ref) then rescale by 1/hb_ref^2.
        for i in range(1, IMAX - 1):
            def flux(level):
                ps = psi[:, level]
                return (0.5 * L * L * K * EPS * F2N2 * np.exp(Z[i] / H0)
                        * np.imag(ps[i] * np.conj(ps[i + 1] + ps[i - 1] - 2.0 * ps[i]) / (DZ * DZ)))
            f1 = flux(0) * inv_hb2
            f2 = flux(1) * inv_hb2
            f3 = flux(2) * inv_hb2
            qy2[i, 3] = qy2[i, 2] + (DT / 12.0) * (5.0 * f1 - 16.0 * f2 + 23.0 * f3)
            # alpha_on=False -> udamp1=udamp2=0

        # ---- invert qy2 -> u2 (per hb^2) with HOMOGENEOUS anomaly BCs ----
        u2 = invert_qy_to_u(qy2[:, 3], lower_bc=0.0)

        # ---- shift time levels ----
        q[:, 0] = q[:, 1]; q[:, 1] = q[:, 2]; q[:, 2] = q[:, 3]
        psi[:, 0] = psi[:, 1]; psi[:, 1] = psi[:, 2]; psi[:, 2] = psi[:, 3]
        qy2[:, 0] = qy2[:, 1]; qy2[:, 1] = qy2[:, 2]; qy2[:, 2] = qy2[:, 3]

        if m % STEPS_PER_DAY == 0:
            psi_time[:, mm] = psi[:, 2]
            qy2_time[:, mm] = qy2[:, 2]
            u2_time[:, mm] = invert_qy_to_u(qy2[:, 2], lower_bc=0.0)
            mm += 1
        m += 1

    return psi_time, qy2_time, u2_time


# ─────────────────────────────────────────────────────────────────────────────
def d2_dz_operators(field2d):
    """Return u2_zz and u2_z (centered interior, one-sided edges) for a (nz,nt)
    array, using the SAME dz as the model."""
    nz, nt = field2d.shape
    uz = np.zeros_like(field2d)
    uzz = np.zeros_like(field2d)
    uz[1:-1, :] = (field2d[2:, :] - field2d[:-2, :]) / (2.0 * DZ)
    uz[0, :] = (field2d[1, :] - field2d[0, :]) / DZ
    uz[-1, :] = (field2d[-1, :] - field2d[-2, :]) / DZ
    uzz[1:-1, :] = (field2d[2:, :] + field2d[:-2, :] - 2.0 * field2d[1:-1, :]) / (DZ * DZ)
    return uz, uzz


def G0_profile():
    """G0(z) = Ucrit(u0) - u0 with the code-consistent Ucrit = betae/(eps K2)."""
    qy0 = qy_of_u(U_BG)
    betae0 = BETA - EPS * qy0
    Ucrit0 = betae0 / (EPS * K2)
    return Ucrit0 - U_BG, Ucrit0, betae0


def G2_field(u2_time):
    """G2 = -u2 - (f2/(N2 K*^2)) [u2_zz - u2_z/H]  (analytic linearization)."""
    u2z, u2zz = d2_dz_operators(u2_time)
    G2 = -u2_time - (F2N2 / KSTAR2) * (u2zz - u2z / H0)
    return G2


def Dmargin_exact(u_field):
    """Exact code-consistent margin Ucrit(U)-U for a full 2-D wind field
    (nz,nt): Ucrit = (beta - eps qy(U))/(eps K2). Uses the same discrete qy."""
    nz, nt = u_field.shape
    D = np.zeros_like(u_field)
    for t in range(nt):
        qy = qy_of_u(u_field[:, t])
        betae = BETA - EPS * qy
        Ucrit = betae / (EPS * K2)
        D[:, t] = Ucrit - u_field[:, t]
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Predicted-boundary machinery
# ─────────────────────────────────────────────────────────────────────────────
BARRIER_Z1_KM = 25.0
BARRIER_Z2_KM = 40.0
WINDOWS_DAYS = [150, 250, 500, 1000]


def _layer_mask():
    zkm = Z / 1000.0
    return (zkm >= BARRIER_Z1_KM) & (zkm <= BARRIER_Z2_KM)


def B_pred_of_hb(G0, G2, hb, window_days):
    """
    Predicted barrier metric:
        D_pred(z,t) = G0(z) + hb^2 G2(z,t)
        B_pred = max_{t<=window} [ min_{25<=z<=40 km} D_pred(z,t) ]
    G2 has shape (nz, nt) sampled daily; window_days truncates t.
    """
    zmask = _layer_mask()
    nt = min(window_days, G2.shape[1])
    Dlayer = G0[zmask, None] + hb * hb * G2[zmask, :nt]   # (nlayer, nt)
    minz = np.min(Dlayer, axis=0)                          # (nt,)
    return float(np.max(minz))


def predicted_hbc(G0, G2, window_days, hb_lo=1.0, hb_hi=200.0, tol=1e-3):
    """
    Smallest hb with B_pred(hb) >= 0, by bisection + a coarse pre-scan (since
    B_pred(hb) is not guaranteed monotonic in general, we scan for the first
    sign change from negative to non-negative on a fine grid, then refine).
    Returns np.nan if B_pred < 0 for all hb up to hb_hi (barrier never opens
    at any tested amplitude within the window).
    """
    hb_grid = np.arange(hb_lo, hb_hi + 0.5, 0.5)
    B = np.array([B_pred_of_hb(G0, G2, hb, window_days) for hb in hb_grid])
    pos = np.where(B >= 0.0)[0]
    if len(pos) == 0:
        return np.nan, hb_grid, B
    first = pos[0]
    if first == 0:
        return float(hb_grid[0]), hb_grid, B
    # refine between hb_grid[first-1] (B<0) and hb_grid[first] (B>=0)
    lo, hi = hb_grid[first - 1], hb_grid[first]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if B_pred_of_hb(G0, G2, mid, window_days) >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi)), hb_grid, B
