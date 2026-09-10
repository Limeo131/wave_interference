"""
scripts/plot_fig04.py
=====================
Generate Figure 4: Peak total wave amplitude A_max(hb, tau) with burst boundary.

Uses the 300-day integration window A_max sweep (fig04_Amax_sweep_300d.npz)
and the v3long burst map (hb_tau_sweep_v3long.npz, with 1000-day boundary verification).

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/plot_fig04.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── global font size settings for publication figures ───────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
})

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR  = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Primary diagnostic altitude
PRIMARY_LEVEL = "32km"
DPI = 600  # publication quality; set to 200 for quick preview

# ── load data ───────────────────────────────────────────────────────────────
# A_max sweep (300-day full integration window)
amax_path = os.path.join(DATA_DIR, "fig04_Amax_sweep_300d.npz")
if not os.path.exists(amax_path):
    raise FileNotFoundError(
        f"A_max sweep data not found: {amax_path}\n"
        "Run: python scripts/run_sweep_fig04.py"
    )

d = np.load(amax_path, allow_pickle=True)
hb_values  = d["hb_values"]       # shape (41,)
tau_values = d["tau_values"]      # shape (61,)
amax_32    = d["amax_32km"]       # shape (41, 61)
amax_25    = d["amax_25km"]
amax_30    = d["amax_30km"]
amax_35    = d["amax_35km"]
amax_40    = d["amax_40km"]

# Burst map from v3long (canonical source with 1000-day boundary verification)
burst_path = os.path.join(DATA_DIR, "hb_tau_sweep_v3long.npz")
if not os.path.exists(burst_path):
    raise FileNotFoundError(f"Burst map not found: {burst_path}")

sw = np.load(burst_path, allow_pickle=True)
burst_map = sw["burst_map"]  # shape (41, 61)

NH, NT = len(hb_values), len(tau_values)
print(f"Loaded A_max sweep (300d): {NH} hb x {NT} tau")
print(f"  Burst count (v3long): {np.sum(burst_map == 1)} / {NH * NT}")
print(f"  A_max at 32km — min: {np.nanmin(amax_32):.3e}, max: {np.nanmax(amax_32):.3e}")

# ── compute optimal critical amplitude ─────────────────────────────────────
valid = np.isfinite(amax_32) & (burst_map >= 0)
af = amax_32[valid]
bf = burst_map[valid]

# Sweep thresholds
threshs = np.linspace(af.min(), af.max(), 5000)
accs = np.array([np.mean((af >= Ac).astype(int) == bf) for Ac in threshs])
best_acc = np.max(accs)

# Take midpoint of accuracy plateau for robustness
plateau = threshs[accs == best_acc]
Ac_final = 0.5 * (plateau.min() + plateau.max())

# Classification metrics
pred = (af >= Ac_final).astype(int)
TP = int(np.sum((pred == 1) & (bf == 1)))
FP = int(np.sum((pred == 1) & (bf == 0)))
FN = int(np.sum((pred == 0) & (bf == 1)))
TN = int(np.sum((pred == 0) & (bf == 0)))
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  Optimal A_c = {Ac_final:.4e} m²/s")
print(f"  Accuracy: {best_acc * 100:.1f}%")
print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"  Precision={precision * 100:.1f}%, Recall={recall * 100:.1f}%, F1={f1:.3f}")

# ── compute burst boundary ─────────────────────────────────────────────────
boundary_hb, boundary_tau = [], []
for j, tau in enumerate(tau_values):
    col = burst_map[:, j]
    bi = np.where(col == 1)[0]
    if len(bi) > 0:
        if bi[0] > 0:
            boundary_hb.append(0.5 * (hb_values[bi[0] - 1] + hb_values[bi[0]]))
        else:
            boundary_hb.append(hb_values[0] - 0.5)
        boundary_tau.append(tau)

# ── identify FN cases ──────────────────────────────────────────────────────
fn_mask = (amax_32 < Ac_final) & (burst_map == 1) & valid
fp_mask = (amax_32 >= Ac_final) & (burst_map == 0) & valid
print(f"\n  False positives: {np.sum(fp_mask)}")
print(f"  False negatives: {np.sum(fn_mask)}")
if np.sum(fn_mask) > 0 and np.sum(fn_mask) <= 20:
    print("  FN cases:")
    for i, j in np.argwhere(fn_mask):
        print(f"    hb={hb_values[i]:.0f}, tau={tau_values[j]:.0f}, "
              f"A_max={amax_32[i, j]:.4e}")


# ============================================================
# Figure 4
# ============================================================
print("\n" + "=" * 60)
print("Generating fig04_peak_total_amplitude.png")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel (a): A_max filled contour + burst boundary + Ac contour ──────────
ax = axes[0]

amax_plot = amax_32.copy()
amax_plot[amax_plot <= 0] = np.nan

# Linear scale (dynamic range ~7x for 300d window)
n_levs = 30
levs = np.linspace(np.nanmin(amax_plot), np.nanmax(amax_plot), n_levs)
cf = ax.contourf(hb_values, tau_values, amax_plot.T,
                 levels=levs, cmap="viridis", extend="both")
cbar = fig.colorbar(cf, ax=ax, shrink=0.92, pad=0.02)
cbar.set_label(r"$A_\mathrm{max}$ (m$^2$ s$^{-1}$)", fontsize=14)

# Critical amplitude contour (red solid)
ax.contour(hb_values, tau_values, amax_plot.T,
           levels=[Ac_final], colors="red", linewidths=2.5, linestyles="-")

# Burst boundary (black dashed — same as fig01)
if boundary_hb:
    ax.plot(boundary_hb, boundary_tau, "k--", linewidth=2.0, zorder=5)

# Format A_c for legend
Ac_mantissa = Ac_final / 1e6
legend_handles = [
    Line2D([0], [0], color="red", lw=2.5, ls="-",
           label=rf"$A_c = {Ac_mantissa:.2f} \times 10^6$"
                 rf" ({best_acc * 100:.1f}%)"),
    Line2D([0], [0], color="k", lw=2.0, ls="--",
           label="Burst boundary"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.9)

ax.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
ax.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
ax.set_title(r"(a) $A_\mathrm{max}(h_b,\tau)$ at $z = 32$ km",
             fontsize=14, loc="left")
ax.set_xlim(hb_values[0] - 0.5, hb_values[-1] + 0.5)
ax.set_ylim(tau_values[0] - 0.5, tau_values[-1] + 0.5)
ax.grid(True, ls=":", alpha=0.3)

# ── Panel (b): classification scatter ──────────────────────────────────────
ax2 = axes[1]
colors_class = {"TP": "#2ca02c", "TN": "#aec6e8", "FP": "#d62728", "FN": "#ff7f0e"}
ms_class = 18

for i in range(NH):
    for j in range(NT):
        if not valid[i, j]:
            continue
        is_burst = burst_map[i, j] == 1
        pred_burst = amax_32[i, j] >= Ac_final
        if is_burst and pred_burst:
            cat = "TP"
        elif not is_burst and not pred_burst:
            cat = "TN"
        elif pred_burst and not is_burst:
            cat = "FP"
        else:
            cat = "FN"
        ax2.scatter(hb_values[i], tau_values[j],
                    c=colors_class[cat], s=ms_class, zorder=3, edgecolors="none")

legend_handles_2 = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["TP"],
           markersize=8, label=f"TP (burst correctly predicted) [{TP}]"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["TN"],
           markersize=8, label=f"TN (no burst correctly predicted) [{TN}]"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["FP"],
           markersize=8, label=f"FP (predicted burst, actual no burst) [{FP}]"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["FN"],
           markersize=8, label=f"FN (predicted no burst, actual burst) [{FN}]"),
]
ax2.legend(handles=legend_handles_2, loc="upper left", fontsize=11, framealpha=0.9)

ax2.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
ax2.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
ax2.set_title(rf"(b) Classification: $A_{{\max}} \gtrless A_c$ "
              rf"(accuracy = {best_acc * 100:.1f}%)",
              fontsize=14, loc="left")
ax2.set_xlim(hb_values[0] - 0.5, hb_values[-1] + 0.5)
ax2.set_ylim(tau_values[0] - 0.5, tau_values[-1] + 0.5)
ax2.grid(True, ls=":", alpha=0.3)

plt.tight_layout()
save_path = os.path.join(FIG_DIR, "fig04_peak_total_amplitude.png")
fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {save_path} (dpi={DPI})")


# ============================================================
# Multi-altitude comparison
# ============================================================
print("\n" + "=" * 60)
print("Multi-altitude A_c comparison (300d window, v3long burst_map)")
print("=" * 60)

for lev_name, amax_lev in [("25km", amax_25), ("30km", amax_30),
                            ("32km", amax_32), ("35km", amax_35),
                            ("40km", amax_40)]:
    v = np.isfinite(amax_lev) & (burst_map >= 0)
    a = amax_lev[v]
    b = burst_map[v]
    th = np.linspace(a.min(), a.max(), 3000)
    acc_arr = np.array([np.mean((a >= t).astype(int) == b) for t in th])
    ba = np.max(acc_arr)
    plat = th[acc_arr == ba]
    bAc = 0.5 * (plat.min() + plat.max())
    print(f"  {lev_name}: best accuracy = {ba * 100:.1f}%, A_c = {bAc:.4e}")

print("\nDone.")
