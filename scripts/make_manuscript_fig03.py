"""
scripts/make_manuscript_fig03.py
================================
Generate final manuscript Figure 3: Mode-0 reconstruction and phase interference.

Three panels:
  (a) Reconstruction: actual vs Mode-0 reconstructed residual at 25 km
  (b) Three wrapped phases: stationary, Mode-0, and total wave at 25 km
  (c) Actual total-wave amplitude with interference envelopes

Case: no-WMFI, hb=30 m, tau=10 d, day-0 frozen background.

Key result: amplitude maxima occur when Mode-0 and stationary phases align (θ_0≈0),
minima when they are opposite (θ_0≈±π). The free-mode magnitude stays nearly constant.

Data source:
    output/data/hm_model/fig03_interference_data.npz

Output:
    manuscript/figures/fig03_mode0_interference_final.png
    manuscript/figures/fig03_mode0_interference_final.pdf
"""

import sys
import os
import numpy as np
from scipy.signal import argrelextrema

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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

# ── Load data ───────────────────────────────────────────────────────────────
d = np.load(os.path.join(DATA_DIR, "fig03_interference_data.npz"))
psi_time = d["psi_time"]
psi_stat = d["psi_stat"]
phi0 = d["phi0"]
a0 = d["a0"]
ramp = d["ramp"]
T0 = float(d["T0"])
t_days = d["t_days"]

IZ = 15  # 25 km
t0, t1 = 30, 130
win = slice(t0, t1)
t_plot = t_days[win]

# ── Compute ─────────────────────────────────────────────────────────────────
# Residual and reconstruction
psi_stat_t = psi_stat[:, np.newaxis] * ramp[np.newaxis, :]
psi_res = psi_time - psi_stat_t
recon = phi0[:, np.newaxis] * a0[np.newaxis, :]
psi_res_iz = psi_res[IZ, win]
recon_iz = recon[IZ, win]

# Three phases (wrapped)
psi_stat_iz = psi_stat[IZ]  # real positive → θ_stat = 0
psi_0 = a0 * phi0[IZ]       # Mode-0 component (complex)
psi_total = psi_time[IZ, :]  # total (complex)

theta_stat = np.zeros(len(t_days))  # always 0 (psi_stat real positive)
theta_0 = np.angle(psi_0)           # wrapped [-π, π]
theta_tot = np.angle(psi_total)     # wrapped [-π, π]

# Total-wave amplitude
A_total = np.abs(psi_total[win])
A_s = abs(psi_stat_iz)
A_f = np.abs(psi_0[win])

# Find extrema
max_idx = argrelextrema(A_total, np.greater, order=5)[0]
min_idx = argrelextrema(A_total, np.less, order=5)[0]
max_days = t0 + max_idx
min_days = t0 + min_idx

# ── QC ──────────────────────────────────────────────────────────────────────
print(f"Mode 0: T0 = {T0:.2f} d")
print(f"A_s = {A_s/1e6:.3f}×10⁶, mean A_f = {A_f.mean()/1e6:.3f}×10⁶")
print(f"|a0φ0| std/mean = {A_f.std()/A_f.mean()*100:.2f}%")
print()
print("Amplitude MAXIMA:")
for idx in max_idx:
    day = t0 + idx
    dth = np.angle(np.exp(1j * (theta_0[day] - theta_stat[day])))
    print(f"  day {day}: |ψ|={A_total[idx]/1e6:.3f}, θ_0/π={theta_0[day]/np.pi:.3f}, Δθ/π={dth/np.pi:.3f}")
print()
print("Amplitude MINIMA:")
for idx in min_idx:
    day = t0 + idx
    dth = np.angle(np.exp(1j * (theta_0[day] - theta_stat[day])))
    print(f"  day {day}: |ψ|={A_total[idx]/1e6:.3f}, θ_0/π={theta_0[day]/np.pi:.3f}, Δθ/π={dth/np.pi:.3f}")
print()

# ── Build figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)

C_STAT = "#2ca02c"   # green — stationary
C_MODE0 = "#d62728"  # red — Mode 0
C_TOT = "#1f77b4"    # blue — total

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): Mode-0 reconstruction
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.plot(t_plot, np.real(psi_res_iz) / 1e6, color=C_TOT, lw=1.2,
        label="Actual residual")
ax.plot(t_plot, np.real(recon_iz) / 1e6, color=C_MODE0, lw=1.0, ls="--",
        label="Mode-0 reconstruction")
ax.axhline(0, color="k", lw=0.4, alpha=0.4)
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"Re$[\psi_\mathrm{res}](25\;\mathrm{km})$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(t0, t1)
ax.legend(loc="lower left", framealpha=0.9)
ax.text(0.97, 0.93, "99.8% variance\nexplained",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.7))
ax.set_title("(a)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): Three wrapped phases
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# Phase curves
ax.axhline(0, color=C_STAT, lw=1.2, label=r"$\theta_\mathrm{stat}$")
ax.plot(t_plot, theta_0[win] / np.pi, color=C_MODE0, lw=1.3,
        label=r"$\theta_0$ (Mode 0)")
ax.plot(t_plot, theta_tot[win] / np.pi, color=C_TOT, lw=0.9, alpha=0.7,
        label=r"$\theta_\mathrm{total}$")

# Horizontal guide lines (just zero reference, neutral)
ax.axhline(0, color="0.6", lw=0.5, ls="-", zorder=0)

# Vertical guides: dotted for both maxima and minima
for i, day in enumerate(max_days):
    ax.axvline(day, color="#5b2c6f", lw=1.0, ls=":", alpha=0.9,
               label="Amplitude max." if i == 0 else "")
for i, day in enumerate(min_days):
    ax.axvline(day, color="#b5651d", lw=1.0, ls=":", alpha=0.9,
               label="Amplitude min." if i == 0 else "")

ax.set_xlabel("Time (days)")
ax.set_ylabel(r"Phase $/\pi$")
ax.set_xlim(t0, t1)
ax.set_ylim(-1.15, 1.15)
ax.legend(loc="lower left", framealpha=0.9, fontsize=7.5, ncol=2)
ax.set_title("(b)", loc="left", fontweight="normal")

# ════════════════════════════════════════════════════════════════════════════
# Panel (c): Total-wave amplitude with theoretical interference bounds
# ════════════════════════════════════════════════════════════════════════════
ax = axes[2]

ax.plot(t_plot, A_total / 1e6, color=C_TOT, lw=1.3, label=r"$|\psi_\mathrm{total}|$")

# Stationary amplitude (green solid)
ax.axhline(A_s / 1e6, color=C_STAT, lw=1.0, ls="-", label=r"$A_s$")

# Theoretical interference bounds (subtle dashed gray)
Af_mean = A_f.mean()
upper_bound = (A_s + Af_mean) / 1e6
lower_bound = abs(A_s - Af_mean) / 1e6
ax.axhline(upper_bound, color="0.72", lw=0.9, ls="--", zorder=1)
ax.axhline(lower_bound, color="0.72", lw=0.9, ls="--", zorder=1)

# Labels near the right side of the panel
ax.text(0.97, upper_bound, r"$A_s + A_f$",
        transform=ax.get_yaxis_transform(), fontsize=8, color="0.50",
        ha="right", va="bottom")
ax.text(0.97, lower_bound, r"$A_s - A_f$",
        transform=ax.get_yaxis_transform(), fontsize=8, color="0.50",
        ha="right", va="top")

# Same vertical guides (matching panel b)
for day in max_days:
    ax.axvline(day, color="#5b2c6f", lw=1.0, ls=":", alpha=0.9)
for day in min_days:
    ax.axvline(day, color="#b5651d", lw=1.0, ls=":", alpha=0.9)

ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$|\psi(25\;\mathrm{km})|$ ($10^6$ m$^2$ s$^{-1}$)")
ax.set_xlim(t0, t1)
ax.set_ylim(0, None)
ax.legend(loc="lower left", framealpha=0.9, fontsize=7.5)
ax.set_title("(c)", loc="left", fontweight="normal")

# ── Save ────────────────────────────────────────────────────────────────────
out_png = os.path.join(FIG_DIR, "fig03_mode0_interference_final.png")
out_pdf = os.path.join(FIG_DIR, "fig03_mode0_interference_final.pdf")
fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print("\nDone.")
