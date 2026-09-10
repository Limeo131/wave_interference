"""
scripts/make_manuscript_fig06.py
================================
Generate final manuscript Figure 6: parameter-space synthesis (unified 1000-day).

Three panels:
  (a) M_max^1000(hb, tau) — propagation-barrier metric with M_max=0 contour
  (b) A_max^1000(hb, tau) — peak total-wave amplitude with A_c contour
  (c) t_open vs t_A — temporal ordering for all 1301 eventual transition cases

All panels use the unified 1000-day integration framework, consistent with Figure 1.
  M_max^1000 > 0 classifies the 1000-day transition map at 2501/2501 (100%).
  A_max^1000 >= A_c classifies the 1000-day transition map at 2501/2501 (100%).
  t_A < t_open for 1301/1301 transition cases (native 1.5-h resolution).

Data source:
    output/data/hm_model/hb_tau_mechanism_1000d.npz

Output:
    manuscript/figures/fig06_parameter_space_synthesis_final.png  (dpi=600)
    manuscript/figures/fig06_parameter_space_synthesis_final.pdf

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/make_manuscript_fig06.py
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# ── Load unified 1000-day data ─────────────────────────────────────────────
data = np.load(os.path.join(DATA_DIR, "hb_tau_mechanism_1000d.npz"))
hb_values = data["hb_values"]
tau_values = data["tau_values"]
burst_1000 = data["burst_1000"]
Mmax_1000 = data["Mmax_1000_legacy"]
Amax_1000 = data["Amax_1000_legacy"]
t_A_native = data["t_A_native"]
t_open_native = data["t_open_native"]
A_C = float(data["A_c"])

# ── Verify ──────────────────────────────────────────────────────────────────
n_trans = int(np.sum(burst_1000 == 1))
n_notrans = int(np.sum(burst_1000 == 0))
assert n_trans == 1301 and n_notrans == 1200
assert np.sum((Mmax_1000 > 0).astype(int) == burst_1000) == 2501
assert np.sum((Amax_1000 >= A_C).astype(int) == burst_1000) == 2501
print(f"Verified: {n_trans} transition, {n_notrans} no-transition")
print(f"Mmax>0 agreement: 2501/2501, Amax>=Ac agreement: 2501/2501")

# ── Compute 1000-day transition boundary ───────────────────────────────────
boundary_hb, boundary_tau = [], []
for j, tau in enumerate(tau_values):
    col = burst_1000[:, j]
    burst_idx = np.where(col == 1)[0]
    if len(burst_idx) > 0:
        first = burst_idx[0]
        if first > 0:
            boundary_hb.append(0.5 * (hb_values[first - 1] + hb_values[first]))
        else:
            boundary_hb.append(hb_values[0] - 0.5)
        boundary_tau.append(tau)

# ── Extract timing for transition cases ─────────────────────────────────────
mask = (burst_1000 == 1)
tA = t_A_native[mask]
tO = t_open_native[mask]
assert np.all(tO > tA), "Timing violation!"
print(f"Timing: 1301/1301 t_A < t_open (native 1.5-h)")

# ── Build figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)

xlim = (hb_values[0] - 1, hb_values[-1] + 1)
ylim = (tau_values[0] - 1, tau_values[-1] + 1)
xticks = np.arange(15, 55, 10)
yticks = np.arange(5, 65, 10)

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): Mmax_1000
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]

Mmax_plot = Mmax_1000 * 1e9  # units: 10^-9 m^-2
vabs = max(abs(np.nanmin(Mmax_plot)), abs(np.nanmax(Mmax_plot)))
norm_m = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
pcm_a = ax.pcolormesh(hb_values, tau_values, Mmax_plot.T,
                      cmap="RdBu_r", norm=norm_m, shading="nearest", rasterized=True)

# M_max = 0 contour
ax.contour(hb_values, tau_values, Mmax_1000.T,
           levels=[0.0], colors="#2ca02c", linewidths=1.2, linestyles="-")

# Wind-based transition boundary (same as Figure 1)
ax.plot(boundary_hb, boundary_tau, "k--", linewidth=1.0, zorder=4)

cbar_a = fig.colorbar(pcm_a, ax=ax, shrink=0.92, pad=0.02, aspect=20)
cbar_a.set_label(r"$M_\mathrm{max}$ ($10^{-9}$ m$^{-2}$)", fontsize=9.5)
cbar_a.ax.tick_params(labelsize=8)

legend_a = [
    Line2D([0], [0], color="#2ca02c", lw=1.2, ls="-", label=r"$M_\mathrm{max}=0$"),
    Line2D([0], [0], color="k", lw=1.0, ls="--", label="Transition boundary"),
]
ax.legend(handles=legend_a, loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor="0.8", borderpad=0.3)

ax.set_xlabel(r"$h_b$ (m)")
ax.set_ylabel(r"$\tau$ (days)")
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_title("(a)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): Amax_1000
# ════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

amax_plot = Amax_1000 / 1e6
Ac_plot = A_C / 1e6

pcm_b = ax2.pcolormesh(hb_values, tau_values, amax_plot.T,
                       cmap="viridis", shading="nearest", rasterized=True)

# A_c contour
ax2.contour(hb_values, tau_values, amax_plot.T,
            levels=[Ac_plot], colors="#d62728", linewidths=1.2, linestyles="-")

# Wind-based transition boundary
ax2.plot(boundary_hb, boundary_tau, "k--", linewidth=1.0, zorder=4)

cbar_b = fig.colorbar(pcm_b, ax=ax2, shrink=0.92, pad=0.02, aspect=20)
cbar_b.set_label(r"$A_\mathrm{max}$ ($10^6$ m$^2$ s$^{-1}$)", fontsize=9.5)
cbar_b.ax.tick_params(labelsize=8)

legend_b = [
    Line2D([0], [0], color="#d62728", lw=1.2, ls="-",
           label=rf"$A_c = {Ac_plot:.2f} \times 10^6$"),
    Line2D([0], [0], color="k", lw=1.0, ls="--", label="Transition boundary"),
]
ax2.legend(handles=legend_b, loc="lower right", fontsize=8,
           framealpha=0.9, edgecolor="0.8", borderpad=0.3)

ax2.set_xlabel(r"$h_b$ (m)")
ax2.set_ylabel(r"$\tau$ (days)")
ax2.set_xlim(xlim); ax2.set_ylim(ylim)
ax2.set_xticks(xticks); ax2.set_yticks(yticks)
ax2.set_title("(b)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (c): Timing — lag plot (Δt = t_open - t_A) vs t_A
# ════════════════════════════════════════════════════════════════════════════
ax3 = axes[2]

lag = tO - tA  # all positive

ax3.scatter(tA, lag, s=4, color="#1f77b4", alpha=0.5, edgecolors="none", zorder=2)

# Zero line for reference (all points should be above)
ax3.axhline(0, color="k", lw=0.6, alpha=0.4, zorder=1)

ax3.text(0.05, 0.92, r"$1301/1301$: $\Delta t > 0$",
         transform=ax3.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

ax3.set_xlabel(r"$t_A$ (days)")
ax3.set_ylabel(r"$\Delta t = t_\mathrm{open} - t_A$ (days)")
ax3.set_xlim(0, tA.max() * 1.05)
ax3.set_ylim(-0.5, lag.max() * 1.15)
ax3.set_title("(c)", loc="left", fontweight="normal")

# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIG_DIR, "fig06_parameter_space_synthesis_final.png")
out_pdf = os.path.join(FIG_DIR, "fig06_parameter_space_synthesis_final.pdf")

fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
print("\nDone.")
