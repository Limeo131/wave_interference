"""
scripts/test_fig5b_critical_wind_margin.py
==========================================
DIAGNOSTIC / TEST ONLY.  Does NOT overwrite the production Fig. 5 and does
NOT modify the manuscript.

Tests a revised Fig. 5b that replaces the propagation-barrier metric
    M(t) = min_{25-40km} m2(z,t; c=0)
with the critical-wind margin
    D_crit(t) = min_{25-40km} [ U_crit(z,t) - u(z,t) ]
where (manuscript definition, c=0, K_*^2 = K^2 - l^2)
    U_crit = [ beta - eps (f0^2/N^2)(u_zz - u_z/H) ] / (eps K_*^2).

Produces (into output/figures/hm_model/fig5b_test/):
  1. fig5b_Dcrit_standalone.{png,pdf}
  2. fig5b_M_vs_Dcrit_comparison.{png,pdf}
  3. fig5_mockup_Dcrit.{png,pdf}   (non-final 4-panel mock-up)

All physical constants / conventions are taken directly from the saved
production runs (sweep_hb30_tau10.00.npz / tau20.00.npz) so this uses the
current production Holton-Mass model outputs and equations.

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/test_fig5b_critical_wind_margin.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "output", "data", "hm_model")
FIGDIR = os.path.join(ROOT, "output", "figures", "hm_model", "fig5b_test")
os.makedirs(FIGDIR, exist_ok=True)

# ── style: mirror production Fig. 5 ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "mathtext.fontset": "cm", "figure.dpi": 150,
    "savefig.dpi": 600, "savefig.bbox": "tight", "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

# ── config (identical to production Fig. 5) ─────────────────────────────────
A_C = 5.76e6
IZ32 = 22
Z1_KM, Z2_KM = 25.0, 40.0
T_WINDOW = 80
H0 = 7000.0
ENSQ = 4.0e-4

C_BURST = "#1f77b4"
C_NOBURST = "#ff7f0e"
C_TA = "#d62728"
C_TOPEN = "#2ca02c"
C_TREV = "#7f7f7f"

# ── event times (native-resolution, same source as production Fig. 5) ───────
mech = np.load(os.path.join(DATA, "hb_tau_mechanism_1000d.npz"))
ih, it = int(30 - 11), int(10 - 1)
t_A = float(mech["t_A_native"][ih, it])       # 29.25
t_open = float(mech["t_open_native"][ih, it]) # 31.375
t_rev = float(mech["t_rev_native"][ih, it])   # 50.0625

d1 = np.load(os.path.join(DATA, "sweep_hb30_tau10.00.npz"))  # transition
d2 = np.load(os.path.join(DATA, "sweep_hb30_tau20.00.npz"))  # no transition

z = d1["z"]; z_km = z / 1000.0
dz = float(d1["dz"]); eps = float(d1["eps"])
k = float(d1["k"]); l = float(d1["l"]); f0 = float(d1["f0"]); beta = float(d1["beta"])
lRm2 = f0**2 / (4.0 * ENSQ * H0**2)
K2 = k**2 + l**2 + lRm2
Ks2 = K2 - l**2

mask = (z_km >= Z1_KM) & (z_km <= Z2_KM)
idx = np.where(mask)[0]
z_layer_km = z_km[idx]


# ── diagnostics ─────────────────────────────────────────────────────────────
def compute_A(d):
    psi = d["psi_time__re"] + 1j * d["psi_time__im"]
    return np.abs(psi[IZ32, :])


def compute_M(d):
    """M(t)=min m2, plus critical-level (u<0 in layer) mask. Matches Fig.5."""
    be = d["betae_time"]; ud = d["ud"]; nt = ud.shape[1]
    M = np.full(nt, np.nan); crit = np.zeros(nt, dtype=bool)
    for j in range(nt):
        ub = ud[idx, j]
        if np.any(ub < 0):
            crit[j] = True
        denom = eps * ub
        m2 = (ENSQ / f0**2) * (be[idx, j] / denom - K2)
        m2[np.abs(denom) < 1e-6] = np.nan
        M[j] = np.nanmin(m2)
    return M, crit


def compute_Dcrit(d):
    """D_crit(t)=min(U_crit-u), z_bottleneck(t), and eps*u>0 layer mask."""
    ud = d["ud"]; nt = ud.shape[1]
    D = np.full(nt, np.nan); zb = np.full(nt, np.nan)
    epsu_ok = np.zeros(nt, dtype=bool)
    for j in range(nt):
        u = ud[:, j]
        u_z = np.gradient(u, dz)
        u_zz = np.gradient(u_z, dz)
        curv = (f0**2 / ENSQ) * (u_zz - u_z / H0)
        Ucrit = (beta - eps * curv) / (eps * Ks2)
        marg = (Ucrit - u)[idx]
        D[j] = np.min(marg)
        zb[j] = z_layer_km[int(np.argmin(marg))]
        epsu_ok[j] = np.all(eps * u[idx] > 0)
    return D, zb, epsu_ok


def compute_Fz(d):
    psi = d["psi_time__re"] + 1j * d["psi_time__im"]
    dpsi_dz = (psi[IZ32 + 1, :] - psi[IZ32 - 1, :]) / (2.0 * dz)
    Fz_raw = -0.5 * k * eps * np.imag(psi[IZ32, :] * np.conj(dpsi_dz))
    return Fz_raw * (f0**2 / (ENSQ * eps))


A1, A2 = compute_A(d1), compute_A(d2)
M1, crit1 = compute_M(d1)
M2, crit2 = compute_M(d2)
D1, zb1, epsu1 = compute_Dcrit(d1)
D2, zb2, epsu2 = compute_Dcrit(d2)
Fz1, Fz2 = compute_Fz(d1), compute_Fz(d2)
u1 = d1["ud"][IZ32, :]; u2 = d2["ud"][IZ32, :]

nt = len(A1)
t_days = np.arange(nt, dtype=float)
tw = T_WINDOW

# For the transition case, D_crit loses physical meaning where eps*u<=0
# anywhere in the layer (the sign equivalence m2>0 <=> U_crit-u>0 requires
# eps*u>0). Mask those days for display and mark the region.
D1_masked = D1.copy()
D1_masked[~epsu1] = np.nan
first_bad1 = np.where(~epsu1[:tw])[0]
t_epsu_bad1 = first_bad1[0] if len(first_bad1) else tw


def add_event_lines(ax):
    ax.axvline(t_A, color=C_TA, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(t_open, color=C_TOPEN, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(t_rev, color=C_TREV, lw=0.8, ls="--", alpha=0.8)


def event_legend(fig, y=-0.04):
    h = [Line2D([0], [0], color=C_TA, lw=0.8, ls="--", label=rf"$t_A={t_A:.1f}$ d"),
         Line2D([0], [0], color=C_TOPEN, lw=0.8, ls="--",
                label=rf"$t_{{\mathrm{{open}}}}={t_open:.1f}$ d"),
         Line2D([0], [0], color=C_TREV, lw=0.8, ls="--",
                label=rf"$t_{{\mathrm{{rev}}}}={t_rev:.1f}$ d")]
    fig.legend(handles=h, loc="lower center", ncol=3, framealpha=0.9,
               edgecolor="0.8", fontsize=8.5, bbox_to_anchor=(0.5, y))


def plot_Dcrit(ax):
    """Shared D_crit(t) panel drawing."""
    # transition case: solid where eps*u>0, gray masked region after
    ax.plot(t_days[:tw], D1_masked[:tw], color=C_BURST, lw=1.2,
            label=r"$\tau=10$ d (transition)")
    ax.plot(t_days[:tw], D2[:tw], color=C_NOBURST, lw=1.2,
            label=r"$\tau=20$ d (no transition)")
    if t_epsu_bad1 < tw:
        ax.axvspan(t_epsu_bad1, tw, color="0.9", zorder=0)
        ax.text(t_epsu_bad1 + 1, ax.get_ylim()[1], r"$\varepsilon\bar u\leq0$",
                fontsize=7, color="0.45", va="top")
    ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
    add_event_lines(ax)
    ax.set_ylabel(r"$D_{\mathrm{crit}}=\min_{25\text{-}40\,\mathrm{km}}"
                  r"(U_{\mathrm{crit}}-\bar u)$ (m s$^{-1}$)")
    ax.set_xlim(0, tw)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: standalone D_crit(t)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
# set a sensible ylim first so axvspan text placement is correct
D1v = D1_masked[:tw]; D2v = D2[:tw]
lo = np.nanmin([np.nanmin(D1v), np.nanmin(D2v)])
hi = np.nanmax([np.nanmax(D1v), np.nanmax(D2v)])
pad = 0.08 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
plot_Dcrit(ax)
ax.legend(loc="lower left", framealpha=0.9)
ax.set_xlabel("Time (days)")
ax.set_title(r"Test: critical-wind margin $D_{\mathrm{crit}}(t)$ "
             r"(candidate replacement for Fig. 5b)", fontsize=10)
event_legend(fig, y=-0.10)
fig.savefig(os.path.join(FIGDIR, "fig5b_Dcrit_standalone.png"))
fig.savefig(os.path.join(FIGDIR, "fig5b_Dcrit_standalone.pdf"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: stacked M(t) vs D_crit(t) comparison
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True,
                         constrained_layout=True)
# (a) M(t)  -- same clipped-display convention as production Fig. 5
axm = axes[0]
M_LO, M_HI = -10.0, 12.0
M1p = M1[:tw] * 1e9
axm.plot(t_days[:tw], np.clip(M1p, M_LO, M_HI), color=C_BURST, lw=1.2,
         label=r"$\tau=10$ d (transition)")
clipped = M1p < M_LO
cd = t_days[:tw][clipped]
if len(cd):
    step = max(1, len(cd) // 6)
    axm.plot(cd[::step], np.full(len(cd[::step]), M_LO + 0.3), "v",
             color=C_BURST, ms=4, alpha=0.6)
axm.plot(t_days[:tw], M2[:tw] * 1e9, color=C_NOBURST, lw=1.2,
         label=r"$\tau=20$ d (no transition)")
first_crit = np.where(crit1[:tw])[0]
tcrit = first_crit[0] if len(first_crit) else tw
if tcrit < tw:
    axm.axvspan(tcrit, tw, color="0.9", zorder=0)
    axm.text(tcrit + 1, M_HI - 1.5, r"$m^2$ ill-conditioned", fontsize=7,
             color="0.45", va="top")
axm.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(axm)
axm.set_ylabel(r"$M(t)$ ($10^{-9}$ m$^{-2}$)")
axm.set_ylim(M_LO, M_HI); axm.set_xlim(0, tw)
axm.set_title("(a)  existing metric $M(t)=\\min_{25\\text{-}40\\,km}\\,m^2$",
              loc="left", fontsize=10)
axm.legend(loc="upper right", framealpha=0.9)
# (b) D_crit(t)
axd = axes[1]
lo = np.nanmin([np.nanmin(D1_masked[:tw]), np.nanmin(D2[:tw])])
hi = np.nanmax([np.nanmax(D1_masked[:tw]), np.nanmax(D2[:tw])])
pad = 0.08 * (hi - lo)
axd.set_ylim(lo - pad, hi + pad)
plot_Dcrit(axd)
axd.set_xlabel("Time (days)")
axd.set_title(r"(b)  candidate $D_{\mathrm{crit}}(t)=\min_{25\text{-}40\,km}"
              r"(U_{\mathrm{crit}}-\bar u)$", loc="left", fontsize=10)
event_legend(fig, y=-0.04)
fig.savefig(os.path.join(FIGDIR, "fig5b_M_vs_Dcrit_comparison.png"))
fig.savefig(os.path.join(FIGDIR, "fig5b_M_vs_Dcrit_comparison.pdf"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: non-final 4-panel mock-up (b -> D_crit)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True,
                         constrained_layout=True)
# (a) amplitude
ax = axes[0, 0]
ax.plot(t_days[:tw], A1[:tw] / 1e6, color=C_BURST, lw=1.2,
        label=r"$\tau=10$ d (transition)")
ax.plot(t_days[:tw], A2[:tw] / 1e6, color=C_NOBURST, lw=1.2,
        label=r"$\tau=20$ d (no transition)")
ax.axhline(A_C / 1e6, color="k", lw=0.7, ls=":", alpha=0.7)
ax.text(tw - 5, A_C / 1e6 + 0.15, r"$A_c$", ha="right", fontsize=9, alpha=0.7)
add_event_lines(ax)
ax.set_ylabel(r"$|\psi(32\;\mathrm{km})|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(0, tw); ax.set_ylim(0, None)
ax.legend(loc="lower right", framealpha=0.9)
ax.set_title("(a)", loc="left")
# (b) D_crit
ax = axes[0, 1]
lo = np.nanmin([np.nanmin(D1_masked[:tw]), np.nanmin(D2[:tw])])
hi = np.nanmax([np.nanmax(D1_masked[:tw]), np.nanmax(D2[:tw])])
pad = 0.08 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
plot_Dcrit(ax)
ax.set_title("(b)", loc="left")
# (c) Fz
ax = axes[1, 0]
ax.plot(t_days[:tw], Fz1[:tw] * 100, color=C_BURST, lw=1.2)
ax.plot(t_days[:tw], Fz2[:tw] * 100, color=C_NOBURST, lw=1.2)
ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(ax)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$F_z(32\;\mathrm{km})$ ($10^{-2}$ m$^2$ s$^{-2}$)")
ax.set_xlim(0, tw)
ax.set_title("(c)", loc="left")
# (d) wind
ax = axes[1, 1]
ax.plot(t_days[:tw], u1[:tw], color=C_BURST, lw=1.2)
ax.plot(t_days[:tw], u2[:tw], color=C_NOBURST, lw=1.2)
ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(ax)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$\bar{u}(32\;\mathrm{km})$ (m s$^{-1}$)")
ax.set_xlim(0, tw)
ax.set_title("(d)", loc="left")
fig.suptitle("NON-FINAL MOCK-UP  (Fig. 5 with panel b = $D_{crit}$)  "
             "- not for publication", fontsize=9, color="0.35")
event_legend(fig, y=-0.05)
fig.savefig(os.path.join(FIGDIR, "fig5_mockup_Dcrit.png"))
fig.savefig(os.path.join(FIGDIR, "fig5_mockup_Dcrit.pdf"))
plt.close(fig)

# ── console summary ─────────────────────────────────────────────────────────
def fzc(t, y):
    y = np.asarray(y, float)
    for i in range(1, len(y)):
        if np.isnan(y[i-1]) or np.isnan(y[i]):
            continue
        if y[i-1] <= 0.0 < y[i]:
            f = (0.0 - y[i-1]) / (y[i] - y[i-1])
            return t[i-1] + f * (t[i] - t[i-1])
    return None

tM = fzc(t_days, M1); tD = fzc(t_days, D1)
print("Figures written to", FIGDIR)
print(f"  t_open from M(t)      = {tM:.4f} d")
print(f"  t_open from D_crit(t) = {tD:.4f} d")
print(f"  |diff|                = {abs(tD - tM):.4f} d")
print(f"  eps*u>0 up to opening = {np.all(epsu1[:int(np.ceil(tM))+1])}")
print(f"  eps*u<=0 first at day = {t_epsu_bad1}")
print(f"  no-transition D_crit max = {np.nanmax(D2):.3f} m/s (opens: {fzc(t_days,D2) is not None})")
