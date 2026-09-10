"""
scripts/make_manuscript_fig04.py
================================
Generate final manuscript Figure 4: free-mode excitation bridge.

Three panels:
  (a) |a0| vs tau — free-mode excitation decreases with forcing spin-up time
      (no-WMFI frozen-background experiments, 14 tau values)
  (b) A_vac vs |a0| — wave-amplitude response vs free-mode excitation
      (full-WMFI at hb=30 m, 12 tau values: 7 transition + 5 no-transition)
  (c) U_vac vs |a0| — mean-flow response vs free-mode excitation

Data sources:
    output/data/hm_model/step2_tau_sweep_recomputed_1000d.npz  (panel a)
    output/data/hm_model/step3_vacillation_diagnostics.npz     (panels b,c)

Definitions:
    |a0|_char: characteristic free-mode projection amplitude (N-weighted, no-WMFI,
               1000-day runs, mean over [max(100,5*tau), 1000] days)
    A_vac: peak-to-peak wave-amplitude vacillation |psi(32km)| in pre-transition window
    U_vac: peak-to-peak zonal-wind vacillation u(32km) in pre-transition window
    Pre-transition window: [day 10, first day u(32km) < 10 m/s]
    For no-transition cases: window extends to end of 300-day run.

Output:
    manuscript/figures/fig04_free_mode_meanflow_bridge_final.png
    manuscript/figures/fig04_free_mode_meanflow_bridge_final.pdf

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/make_manuscript_fig04.py
"""

import sys
import os
import numpy as np
from scipy import stats

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

# ── Publication figure style ────────────────────────────────────────────────
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

# ── Colors ──────────────────────────────────────────────────────────────────
C_MAIN = "#1f77b4"
C_BURST = "#d62728"
C_NOBURST = "#1f77b4"
C_FIT = "0.4"

# ── Load data ───────────────────────────────────────────────────────────────
# Panel (a): no-WMFI tau sweep (recomputed, 1000-day, fully reproducible)
d_tau = np.load(os.path.join(DATA_DIR, "step2_tau_sweep_recomputed_1000d.npz"))
tau_a = d_tau["tau_values"]       # (14,)
a0_a = d_tau["a0_char"]          # (14,)
psi_stat_max_val = float(d_tau["psi_stat_max"])  # single tau-independent value

# Panels (b,c): full-WMFI bridge diagnostics
d_vac = np.load(os.path.join(DATA_DIR, "step3_vacillation_diagnostics.npz"))
tau_bc = d_vac["tau_values"]     # (12,)
burst = d_vac["burst"]           # (12,) bool
A_vac = d_vac["A_vac_ptp"]      # (12,)
U_vac = d_vac["U_vac_ptp"]      # (12,)

# Replace the old a0_steady horizontal coordinate in panels (b,c)
# with the recomputed a0_char for the 12 shared tau values
a0_bc = np.array([a0_a[list(tau_a).index(t)] for t in tau_bc])

noburst = ~burst
n_burst = int(np.sum(burst))
n_noburst = int(np.sum(noburst))

# ── QC ──────────────────────────────────────────────────────────────────────
print(f"Panel (a): {len(tau_a)} tau values, range [{tau_a.min():.0f}, {tau_a.max():.0f}] d")
print(f"  |a0| range: [{a0_a.min():.3e}, {a0_a.max():.3e}]")
print(f"  Monotonically decreasing: {np.all(np.diff(a0_a) < 0)}")
rho_a, _ = stats.spearmanr(tau_a, a0_a)
print(f"  Spearman r(tau, |a0|) = {rho_a:.4f}")
print()

print(f"Panels (b,c): {len(tau_bc)} cases at hb=30 m")
print(f"  {n_burst} transition (tau={tau_bc[burst].tolist()})")
print(f"  {n_noburst} no-transition (tau={tau_bc[noburst].tolist()})")
print()

# Panel (b) correlations
r_b_nb, p_b_nb = stats.pearsonr(a0_bc[noburst], A_vac[noburst])
rho_b_nb, _ = stats.spearmanr(a0_bc[noburst], A_vac[noburst])
r_b_all, p_b_all = stats.pearsonr(a0_bc, A_vac)
print(f"Panel (b): A_vac vs |a0|")
print(f"  No-transition: r={r_b_nb:.4f}, p={p_b_nb:.2e}, N={n_noburst}")
print(f"  All cases: r={r_b_all:.4f}, p={p_b_all:.2e}")
print()

# Panel (c) correlations
r_c_nb, p_c_nb = stats.pearsonr(a0_bc[noburst], U_vac[noburst])
rho_c_nb, _ = stats.spearmanr(a0_bc[noburst], U_vac[noburst])
r_c_all, p_c_all = stats.pearsonr(a0_bc, U_vac)
print(f"Panel (c): U_vac vs |a0|")
print(f"  No-transition: r={r_c_nb:.4f}, p={p_c_nb:.2e}, N={n_noburst}")
print(f"  All cases: r={r_c_all:.4f}, p={p_c_all:.2e}")
print(f"  U_vac transition range: [{U_vac[burst].min():.1f}, {U_vac[burst].max():.1f}] m/s")
print(f"  U_vac no-transition range: [{U_vac[noburst].min():.1f}, {U_vac[noburst].max():.1f}] m/s")
print()

# ── Build figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): |a0| vs tau
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]

ax.plot(tau_a, a0_a / 1e6, "o-", color=C_MAIN, ms=5, lw=1.2, markeredgewidth=0)

# Stationary-wave reference line
ax.axhline(psi_stat_max_val / 1e6, color="0.5", lw=0.8, ls="--", alpha=0.7)
ax.text(85, psi_stat_max_val / 1e6 + 0.08, r"$|\psi_\mathrm{stat}|_\mathrm{max}$",
        fontsize=8, color="0.4", va="bottom")

ax.set_xlabel(r"$\tau$ (days)")
ax.set_ylabel(r"$|a_0|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(0, 105)
ax.set_ylim(0, 4.5)
ax.set_title("(a)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): A_vac vs |a0|
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# No-transition (filled)
ax.scatter(a0_bc[noburst] / 1e6, A_vac[noburst] / 1e6, s=40,
           marker="o", c=C_NOBURST, edgecolors="none", zorder=3,
           label="No transition")
# Transition (open)
ax.scatter(a0_bc[burst] / 1e6, A_vac[burst] / 1e6, s=40,
           marker="o", facecolors="none", edgecolors=C_BURST, linewidths=1.2,
           zorder=3, label="Transition")

# Fit line (no-transition only)
slope_b, intercept_b, _, _, _ = stats.linregress(a0_bc[noburst], A_vac[noburst])
x_fit = np.linspace(0, a0_bc.max() * 1.1, 50)
ax.plot(x_fit / 1e6, (slope_b * x_fit + intercept_b) / 1e6,
        color=C_FIT, lw=0.8, ls="--", alpha=0.7)

ax.text(0.95, 0.08, rf"$r = {r_b_nb:.2f}$" + "\n(no trans.)",
        transform=ax.transAxes, fontsize=8, ha="right", color=C_FIT)

ax.set_xlabel(r"$|a_0|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_ylabel(r"$A_\mathrm{vac}$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(0, 4.2)
ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
ax.set_title("(b)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (c): U_vac vs |a0|
# ════════════════════════════════════════════════════════════════════════════
ax = axes[2]

# No-transition (filled)
ax.scatter(a0_bc[noburst] / 1e6, U_vac[noburst], s=40,
           marker="o", c=C_NOBURST, edgecolors="none", zorder=3)
# Transition (open)
ax.scatter(a0_bc[burst] / 1e6, U_vac[burst], s=40,
           marker="o", facecolors="none", edgecolors=C_BURST, linewidths=1.2,
           zorder=3)

# Fit line (no-transition only)
slope_c, intercept_c, _, _, _ = stats.linregress(a0_bc[noburst], U_vac[noburst])
ax.plot(x_fit / 1e6, slope_c * x_fit + intercept_c,
        color=C_FIT, lw=0.8, ls="--", alpha=0.7)

ax.text(0.95, 0.08, rf"$r = {r_c_nb:.2f}$" + "\n(no trans.)",
        transform=ax.transAxes, fontsize=8, ha="right", color=C_FIT)

ax.set_xlabel(r"$|a_0|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_ylabel(r"$U_\mathrm{vac}$ (m s$^{-1}$)")
ax.set_xlim(0, 4.2)
ax.set_title("(c)", loc="left", fontweight="normal")

# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIG_DIR, "fig04_free_mode_meanflow_bridge_final.png")
out_pdf = os.path.join(FIG_DIR, "fig04_free_mode_meanflow_bridge_final.pdf")

fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print("\nDone.")
