"""
scripts/make_manuscript_fig02.py
================================
Generate final manuscript Figure 2: propagation geometry and vertical structures.

Two panels:
  (a) Refractive-index profiles m²(z) for the stationary wave (c=0) and Mode 0 (c=c0)
  (b) Normalized vertical amplitude structures of the stationary response and Mode 0

All diagnostics use the day-0 frozen background wind U0(z).

Data source:
    output/data/hm_model/fig02_propagation_data.npz
    (precomputed from production eigensolver and iterative stationary solver)

Output:
    manuscript/figures/fig02_propagation_vertical_structure_final.png
    manuscript/figures/fig02_propagation_vertical_structure_final.pdf

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/make_manuscript_fig02.py
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "legend.fontsize": 9,
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

# ── Load precomputed data ───────────────────────────────────────────────────
d = np.load(os.path.join(DATA_DIR, "fig02_propagation_data.npz"))
z_km = d["z_km"]
m2_stat = d["m2_stat"]
m2_mode0 = d["m2_mode0"]
psi_stat_norm = d["psi_stat_norm"]
phi0_norm = d["phi0_norm"]
z_turn_stat = float(d["z_turn_stat"])
z_turn_mode0 = float(d["z_turn_mode0"])
T0 = float(d["T0"])
c0 = float(d["c0"])

# ── QC ──────────────────────────────────────────────────────────────────────
print(f"Mode 0: T0 = {T0:.2f} d, c0 = {c0:.3f} m/s")
print(f"Turning levels: stat = {z_turn_stat:.2f} km, Mode 0 = {z_turn_mode0:.2f} km")
print(f"  difference: {z_turn_mode0 - z_turn_stat:.2f} km")

# ── Colors ──────────────────────────────────────────────────────────────────
C_STAT = "#1f77b4"    # blue — stationary wave
C_MODE0 = "#d62728"   # red — Mode 0

# ── Build figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5), sharey=True,
                         constrained_layout=True)

# Altitude range
ylim = (10, 50)

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): Refractive-index profiles
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Scale m² to 10⁻⁸ m⁻² for readable axis
ax.plot(m2_stat * 1e8, z_km, color=C_STAT, lw=1.5,
        label=r"Stationary ($c=0$)")
ax.plot(m2_mode0 * 1e8, z_km, color=C_MODE0, lw=1.5, ls="--",
        label=rf"Mode 0 ($c={c0:.1f}$ m s$^{{-1}}$)")

# m²=0 reference
ax.axvline(0, color="k", lw=0.6, alpha=0.5)

# Turning-level markers
ax.axhline(z_turn_stat, color=C_STAT, lw=0.7, ls=":", alpha=0.7)
ax.axhline(z_turn_mode0, color=C_MODE0, lw=0.7, ls=":", alpha=0.7)

# Annotations
ax.text(0.15, z_turn_stat + 0.4, f"{z_turn_stat:.1f} km",
        fontsize=8, color=C_STAT, va="bottom")
ax.text(0.15, z_turn_mode0 + 0.4, f"{z_turn_mode0:.1f} km",
        fontsize=8, color=C_MODE0, va="bottom")

ax.set_xlabel(r"$m^2$ ($10^{-8}$ m$^{-2}$)")
ax.set_ylabel(r"Altitude (km)")
ax.set_xlim(-2.5, 3.0)
ax.set_ylim(ylim)
ax.legend(loc="upper right", framealpha=0.95)
ax.set_title("(a)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): Normalized vertical structures
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]

ax.plot(psi_stat_norm, z_km, color=C_STAT, lw=1.5,
        label=r"$|\psi_\mathrm{stat}|$")
ax.plot(phi0_norm, z_km, color=C_MODE0, lw=1.5, ls="--",
        label=r"$|\phi_0|$ (Mode 0)")

# Turning-level markers
ax.axhline(z_turn_stat, color=C_STAT, lw=0.7, ls=":", alpha=0.7)
ax.axhline(z_turn_mode0, color=C_MODE0, lw=0.7, ls=":", alpha=0.7)

ax.set_xlabel("Normalized amplitude")
ax.set_xlim(0, 1.05)
ax.set_ylim(ylim)
ax.legend(loc="upper right", framealpha=0.95)
ax.set_title("(b)", loc="left", fontweight="normal")

# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIG_DIR, "fig02_propagation_vertical_structure_final.png")
out_pdf = os.path.join(FIG_DIR, "fig02_propagation_vertical_structure_final.pdf")

fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
print("\nDone.")
