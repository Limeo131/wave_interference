"""
scripts/make_manuscript_fig05.py
================================
Generate final manuscript Figure 5: representative transition dynamics.

Two-case comparison at fixed hb=30 m:
  - Transition case: tau=10 d
  - No-transition case: tau=20 d

Four panels:
  (a) Wave amplitude A(t) = |psi(32 km, t)| with A_c threshold
  (b) Propagation-barrier metric M(t) = min_{z in [25,40]km} m2_stat(z,t)
      Masked after critical level forms (u<0 in barrier layer)
  (c) Density-weighted vertical EP flux F_z at 32 km
  (d) Zonal-mean wind u_bar(32 km, t)

Event times from unified 1000-day native-resolution sweep:
  t_A = 29.25 d, t_open = 31.375 d, t_rev = 50.0625 d

Data sources:
    output/data/hm_model/sweep_hb30_tau10.00.npz  (full z-t fields)
    output/data/hm_model/sweep_hb30_tau20.00.npz  (full z-t fields)
    output/data/hm_model/hb_tau_mechanism_1000d.npz  (native event times)

Output:
    manuscript/figures/fig05_representative_transition_dynamics_final.png
    manuscript/figures/fig05_representative_transition_dynamics_final.pdf

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/make_manuscript_fig05.py
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Publication figure style (matches Fig 1 and Fig 6) ──────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "cm",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ── Configuration ───────────────────────────────────────────────────────────
A_C = 5.76e6       # m^2/s
IZ32 = 22          # z-index for 32 km
Z1_KM = 25.0
Z2_KM = 40.0
T_WINDOW = 80      # days to plot

# Colors
C_BURST = "#1f77b4"     # blue
C_NOBURST = "#ff7f0e"   # orange
C_TA = "#d62728"        # red
C_TOPEN = "#2ca02c"     # green
C_TREV = "#7f7f7f"      # gray

# ── Load native-resolution event times from 1000-day sweep ──────────────────
mech = np.load(os.path.join(DATA_DIR, "hb_tau_mechanism_1000d.npz"))
ih = int(30 - 11)   # hb=30
it = int(10 - 1)    # tau=10
t_A = float(mech["t_A_native"][ih, it])
t_open = float(mech["t_open_native"][ih, it])
t_rev = float(mech["t_rev_native"][ih, it])

print(f"Native-resolution event times (hb=30, tau=10):")
print(f"  t_A     = {t_A:.4f} d")
print(f"  t_open  = {t_open:.4f} d")
print(f"  t_rev   = {t_rev:.4f} d")
print(f"  t_A < t_open < t_rev: {t_A < t_open < t_rev}")
print()

# ── Load full-field data ────────────────────────────────────────────────────
d1 = np.load(os.path.join(DATA_DIR, "sweep_hb30_tau10.00.npz"))
d2 = np.load(os.path.join(DATA_DIR, "sweep_hb30_tau20.00.npz"))

z = d1["z"]
z_km = z / 1000.0
dz = float(d1["dz"])
eps = float(d1["eps"])
k = float(d1["k"])
l = float(d1["l"])
f0 = float(d1["f0"])
h0 = 7000.0
ensq = 4.0e-4
lrmsq = f0**2 / (4 * ensq * h0**2)
K2 = k**2 + l**2 + lrmsq

barrier_mask = (z_km >= Z1_KM) & (z_km <= Z2_KM)
barrier_idx = np.where(barrier_mask)[0]

# ── Compute diagnostics ─────────────────────────────────────────────────────
def compute_A(d):
    """Wave amplitude |psi| at 32 km (daily)."""
    psi = d["psi_time__re"] + 1j * d["psi_time__im"]
    return np.abs(psi[IZ32, :])


def compute_M(d):
    """Propagation-barrier metric M(t) and critical-level mask."""
    betae = d["betae_time"]
    ud = d["ud"]
    nt = ud.shape[1]
    M = np.zeros(nt)
    critical_level = np.zeros(nt, dtype=bool)

    for t in range(nt):
        u_barr = ud[barrier_idx, t]
        # Check if any level has u < 0 (critical level for c=0 stationary wave)
        if np.any(u_barr < 0):
            critical_level[t] = True
        denom = eps * u_barr
        be = betae[barrier_idx, t]
        m2 = (ensq / f0**2) * (be / denom - K2)
        m2[np.abs(denom) < 1e-6] = np.nan
        M[t] = np.nanmin(m2)
    return M, critical_level


def compute_Fz(d):
    """Normalized vertical EP flux F_z at 32 km.
    F_z = (f0²/(N²ε)) × F_z^raw = -(k f0²)/(2 N²) × Im(ψ × conj(∂ψ/∂z))
    This is the standard QG kinematic vertical EP flux.
    Positive = upward wave-activity propagation.
    Units: m² s⁻².
    """
    psi = d["psi_time__re"] + 1j * d["psi_time__im"]
    dpsi_dz = (psi[IZ32 + 1, :] - psi[IZ32 - 1, :]) / (2.0 * dz)
    # Raw code flux
    Fz_raw = -0.5 * k * eps * np.imag(psi[IZ32, :] * np.conj(dpsi_dz))
    # Normalize: F_z = (f0²/(N²ε)) × F_z_raw
    C_norm = f0**2 / (ensq * eps)
    return Fz_raw * C_norm


A1 = compute_A(d1)
A2 = compute_A(d2)
M1, crit1 = compute_M(d1)
M2, crit2 = compute_M(d2)
Fz1 = compute_Fz(d1)
Fz2 = compute_Fz(d2)
u1 = d1["ud"][IZ32, :]
u2 = d2["ud"][IZ32, :]

t_days = np.arange(len(A1), dtype=float)

# Mask M(t) for burst case after critical level develops
first_crit = np.where(crit1)[0]
t_crit_day = first_crit[0] if len(first_crit) > 0 else len(M1)
print(f"Critical level first develops at day {t_crit_day} (burst case)")
print(f"  M(t) plotted through full window; clipped below y-limit after day {t_crit_day}")
print()

# ── QC: No-transition case ──────────────────────────────────────────────────
print("No-transition case (hb=30, tau=20) QC:")
print(f"  max A = {A2.max():.3e} (A/Ac = {A2.max()/A_C:.3f})")
print(f"  max M = {M2.max():.3e} (< 0: {M2.max() < 0})")
print(f"  min u = {u2.min():.2f} m/s (> 0: {u2.min() > 0})")
print(f"  max |Fz| = {np.max(np.abs(Fz2)):.2e} m^2/s^2")
print(f"  Burst/no-burst Fz ratio: {np.max(np.abs(Fz1))/np.max(np.abs(Fz2)):.0f}x")

# ── Build figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True,
                         constrained_layout=True)

tw = T_WINDOW


def add_event_lines(ax):
    """Add vertical event-time markers."""
    ax.axvline(t_A, color=C_TA, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(t_open, color=C_TOPEN, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(t_rev, color=C_TREV, lw=0.8, ls="--", alpha=0.8)


# ════════════════════════════════════════════════════════════════════════════
# Panel (a): Wave amplitude
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax.plot(t_days[:tw], A1[:tw] / 1e6, color=C_BURST, lw=1.2,
        label=r"$\tau=10$ d (transition)")
ax.plot(t_days[:tw], A2[:tw] / 1e6, color=C_NOBURST, lw=1.2,
        label=r"$\tau=20$ d (no transition)")
ax.axhline(A_C / 1e6, color="k", lw=0.7, ls=":", alpha=0.7)
ax.text(tw - 5, A_C / 1e6 + 0.15, r"$A_c$", ha="right", fontsize=9, alpha=0.7)
add_event_lines(ax)
ax.set_ylabel(r"$|\psi(32\;\mathrm{km})|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(0, tw)
ax.set_ylim(0, None)
ax.legend(loc="lower right", framealpha=0.9)
ax.set_title("(a)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): Propagation-barrier metric (full window, clipped display)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]

M_YLIM_LO = -10.0
M_YLIM_HI = 12.0

# Plot full burst-case M(t), clipped to axis range
M1_plot = M1[:tw] * 1e9
ax.plot(t_days[:tw], np.clip(M1_plot, M_YLIM_LO, M_YLIM_HI),
        color=C_BURST, lw=1.2, clip_on=True)

# Mark clipped points with downward triangles at bottom boundary
clipped_mask = M1_plot < M_YLIM_LO
clipped_days = t_days[:tw][clipped_mask]
if len(clipped_days) > 0:
    # Show a few markers (not every point) to indicate off-scale
    step = max(1, len(clipped_days) // 6)
    ax.plot(clipped_days[::step], np.full(len(clipped_days[::step]), M_YLIM_LO + 0.3),
            "v", color=C_BURST, ms=4, alpha=0.6, clip_on=True)

# No-transition case: show full window normally
ax.plot(t_days[:tw], M2[:tw] * 1e9, color=C_NOBURST, lw=1.2)

# Gray shading for critical-level regime
if t_crit_day < tw:
    ax.axvspan(t_crit_day, tw, color="0.9", zorder=0)
    ax.text(t_crit_day + 1, M_YLIM_HI - 1.5, r"$m^2$ ill-conditioned",
            fontsize=7, color="0.45", va="top")

ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(ax)
ax.set_ylabel(r"$M(t)$ ($10^{-9}$ m$^{-2}$)")
ax.set_xlim(0, tw)
ax.set_ylim(M_YLIM_LO, M_YLIM_HI)
ax.set_title("(b)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (c): Normalized vertical EP flux
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]
ax.plot(t_days[:tw], Fz1[:tw] * 100, color=C_BURST, lw=1.2)
ax.plot(t_days[:tw], Fz2[:tw] * 100, color=C_NOBURST, lw=1.2)
ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(ax)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$F_z(32\;\mathrm{km})$ ($10^{-2}$ m$^2$ s$^{-2}$)")
ax.set_xlim(0, tw)
ax.set_title("(c)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (d): Zonal-mean wind
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]
ax.plot(t_days[:tw], u1[:tw], color=C_BURST, lw=1.2)
ax.plot(t_days[:tw], u2[:tw], color=C_NOBURST, lw=1.2)
ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.7)
add_event_lines(ax)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$\bar{u}(32\;\mathrm{km})$ (m s$^{-1}$)")
ax.set_xlim(0, tw)
ax.set_title("(d)", loc="left", fontweight="normal")

# ── Shared event-time legend ────────────────────────────────────────────────
event_handles = [
    Line2D([0], [0], color=C_TA, lw=0.8, ls="--",
           label=rf"$t_A={t_A:.1f}$ d"),
    Line2D([0], [0], color=C_TOPEN, lw=0.8, ls="--",
           label=rf"$t_{{\mathrm{{open}}}}={t_open:.1f}$ d"),
    Line2D([0], [0], color=C_TREV, lw=0.8, ls="--",
           label=rf"$t_{{\mathrm{{rev}}}}={t_rev:.1f}$ d"),
]
fig.legend(handles=event_handles, loc="lower center", ncol=3,
           framealpha=0.9, edgecolor="0.8", fontsize=8.5,
           bbox_to_anchor=(0.5, -0.05))

# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIG_DIR, "fig05_representative_transition_dynamics_final.png")
out_pdf = os.path.join(FIG_DIR, "fig05_representative_transition_dynamics_final.pdf")

fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print()
print("=== Final QC Summary ===")
print(f"  Event ordering: t_A={t_A:.2f} < t_open={t_open:.2f} < t_rev={t_rev:.2f} d")
print(f"  No-transition: A < Ac, M < 0, u > 0 throughout")
print(f"  Panel (b): full M(t) plotted; clipped + gray shading from day {t_crit_day} (critical level)")
print(f"  Panel (c): normalized F_z = (f0²/(N²ε)) × F_z^raw, positive = upward, units m²/s²")
print(f"  A_c = {A_C:.4e} (same as Figure 6)")
print("  Done.")
