"""
scripts/plot_fig04_sensitivity.py
=================================
Plot the Figure 4 sensitivity suite from pre-computed data.

Reads:
  output/data/hm_model/fig04_Amax_v1_1000d.npz   (v1: full-run, boundary at 1000d)
  output/data/hm_model/fig04_sensitivity.npz      (v2/v3/v4: pre-event variants)
  output/data/hm_model/hb_tau_sweep_v3long.npz    (canonical burst map)

Produces:
  output/figures/hm_model/fig4/fig04_v1_full.png
  output/figures/hm_model/fig4/fig04_v2_pre_u0.png
  output/figures/hm_model/fig4/fig04_v3_pre_u10.png
  output/figures/hm_model/fig4/fig04_v4_pre_decel.png
  output/figures/hm_model/fig4/fig04_comparison_table.txt

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/plot_fig04_sensitivity.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR  = os.path.join(ROOT, "output", "figures", "hm_model", "fig4")
os.makedirs(FIG_DIR, exist_ok=True)

# ── global font settings ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
})

DPI = 200  # set to 600 for publication

# ── load data ───────────────────────────────────────────────────────────────
# v1: full-run max (boundary cases at 1000d)
v1_path = os.path.join(DATA_DIR, "fig04_Amax_v1_1000d.npz")
if not os.path.exists(v1_path):
    raise FileNotFoundError(f"v1 data not found: {v1_path}\nRun: python scripts/run_fig04_v1_boundary.py")
d_v1 = np.load(v1_path)
hb_values  = d_v1["hb_values"]
tau_values = d_v1["tau_values"]
amax_v1    = d_v1["amax_32km"]

# v2/v3/v4: pre-event variants
sens_path = os.path.join(DATA_DIR, "fig04_sensitivity.npz")
if not os.path.exists(sens_path):
    raise FileNotFoundError(f"Sensitivity data not found: {sens_path}\nRun: python scripts/run_fig04_sensitivity.py")
d_sens = np.load(sens_path)
amax_v2 = d_sens["amax_pre_u0"]
amax_v3 = d_sens["amax_pre_u10"]
amax_v4 = d_sens["amax_pre_decel"]

# Canonical burst map
sw = np.load(os.path.join(DATA_DIR, "hb_tau_sweep_v3long.npz"))
burst_ref = sw["burst_map"]

NH, NT = len(hb_values), len(tau_values)
print(f"Grid: {NH} x {NT}, burst count: {np.sum(burst_ref == 1)}/{burst_ref.size}")

# ── burst boundary ─────────────────────────────────────────────────────────
boundary_hb, boundary_tau = [], []
for j, tau in enumerate(tau_values):
    col = burst_ref[:, j]
    bi = np.where(col == 1)[0]
    if len(bi) > 0:
        if bi[0] > 0:
            boundary_hb.append(0.5 * (hb_values[bi[0] - 1] + hb_values[bi[0]]))
        else:
            boundary_hb.append(hb_values[0] - 0.5)
        boundary_tau.append(tau)


# ── classification + plotting function ─────────────────────────────────────
def classify_and_plot(amax_map, var_key, var_title, save_name):
    """Find optimal Ac, compute metrics, plot 2-panel figure."""
    valid = np.isfinite(amax_map) & (burst_ref >= 0)
    af = amax_map[valid]
    bf = burst_ref[valid]

    # Optimal threshold
    threshs = np.linspace(af.min(), af.max(), 5000)
    accs = np.array([np.mean((af >= Ac).astype(int) == bf) for Ac in threshs])
    best_acc = np.max(accs)
    plateau = threshs[accs == best_acc]
    Ac = 0.5 * (plateau.min() + plateau.max())

    pred = (af >= Ac).astype(int)
    TP = int(np.sum((pred == 1) & (bf == 1)))
    FP = int(np.sum((pred == 1) & (bf == 0)))
    FN = int(np.sum((pred == 0) & (bf == 1)))
    TN = int(np.sum((pred == 0) & (bf == 0)))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  {var_key}: Ac={Ac:.4e}, acc={best_acc*100:.1f}%, "
          f"TP={TP} TN={TN} FP={FP} FN={FN}, F1={f1:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): contour
    ax = axes[0]
    amax_plot = amax_map.copy()
    amax_plot[amax_plot <= 0] = np.nan
    levs = np.linspace(np.nanmin(amax_plot), np.nanmax(amax_plot), 30)
    cf = ax.contourf(hb_values, tau_values, amax_plot.T,
                     levels=levs, cmap="viridis", extend="both")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label(r"$A_\mathrm{max}$ (m$^2$ s$^{-1}$)", fontsize=14)
    ax.contour(hb_values, tau_values, amax_plot.T,
               levels=[Ac], colors="red", linewidths=2.5, linestyles="-")
    if boundary_hb:
        ax.plot(boundary_hb, boundary_tau, "k--", linewidth=2.0, zorder=5)

    Ac_m = Ac / 1e6
    legend_handles = [
        Line2D([0], [0], color="red", lw=2.5, ls="-",
               label=rf"$A_c = {Ac_m:.2f} \times 10^6$ ({best_acc*100:.1f}%)"),
        Line2D([0], [0], color="k", lw=2.0, ls="--",
               label="Burst boundary"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.9)
    ax.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
    ax.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
    ax.set_title(f"(a) {var_title}", fontsize=14, loc="left")
    ax.set_xlim(hb_values[0] - 0.5, hb_values[-1] + 0.5)
    ax.set_ylim(tau_values[0] - 0.5, tau_values[-1] + 0.5)
    ax.grid(True, ls=":", alpha=0.3)

    # Panel (b): classification scatter
    ax2 = axes[1]
    colors_class = {"TP": "#2ca02c", "TN": "#aec6e8", "FP": "#d62728", "FN": "#ff7f0e"}
    ms = 18
    for i in range(NH):
        for j in range(NT):
            if not valid[i, j]:
                continue
            is_burst = burst_ref[i, j] == 1
            pred_burst = amax_map[i, j] >= Ac
            if is_burst and pred_burst:
                cat = "TP"
            elif not is_burst and not pred_burst:
                cat = "TN"
            elif pred_burst and not is_burst:
                cat = "FP"
            else:
                cat = "FN"
            ax2.scatter(hb_values[i], tau_values[j],
                        c=colors_class[cat], s=ms, zorder=3, edgecolors="none")

    legend_handles_2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["TP"],
               markersize=8, label=f"TP [{TP}]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["TN"],
               markersize=8, label=f"TN [{TN}]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["FP"],
               markersize=8, label=f"FP [{FP}]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_class["FN"],
               markersize=8, label=f"FN [{FN}]"),
    ]
    ax2.legend(handles=legend_handles_2, loc="upper left", fontsize=11, framealpha=0.9)
    ax2.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
    ax2.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
    ax2.set_title(f"(b) Classification (acc={best_acc*100:.1f}%)",
                  fontsize=14, loc="left")
    ax2.set_xlim(hb_values[0] - 0.5, hb_values[-1] + 0.5)
    ax2.set_ylim(tau_values[0] - 0.5, tau_values[-1] + 0.5)
    ax2.grid(True, ls=":", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, save_name)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {
        "variant": var_key, "title": var_title,
        "Ac": Ac, "accuracy": best_acc,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ── generate all figures ───────────────────────────────────────────────────
print("=" * 60)
print("Generating Figure 4 sensitivity suite")
print("=" * 60)

results = []

# v1: full-run (1000d boundary)
results.append(classify_and_plot(
    amax_v1, "v1_full",
    "Full-run maximum (1000d boundary)",
    "fig04_v1_full.png",
))

# v2: pre-burst u < 0
results.append(classify_and_plot(
    amax_v2, "v2_pre_u0",
    r"Pre-burst ($\bar{u} < 0$)",
    "fig04_v2_pre_u0.png",
))

# v3: pre-burst u < 10
results.append(classify_and_plot(
    amax_v3, "v3_pre_u10",
    r"Pre-burst ($\bar{u} < 10$ m/s)",
    "fig04_v3_pre_u10.png",
))

# v4: pre-deceleration
results.append(classify_and_plot(
    amax_v4, "v4_pre_decel",
    r"Pre-decel (5d-smooth $d\bar{u}/dt < -1$)",
    "fig04_v4_pre_decel.png",
))

# ── comparison table ───────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("COMPARISON SUMMARY TABLE")
print("=" * 80)
header = (f"{'Variant':<16} {'Ac (x10^6)':>10} {'Accuracy':>9} "
          f"{'TP':>5} {'TN':>5} {'FP':>4} {'FN':>4} "
          f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
print(header)
print("-" * len(header))
for r in results:
    line = (f"{r['variant']:<16} {r['Ac']/1e6:>10.3f} {r['accuracy']*100:>8.1f}% "
            f"{r['TP']:>5} {r['TN']:>5} {r['FP']:>4} {r['FN']:>4} "
            f"{r['precision']*100:>5.1f}% {r['recall']*100:>6.1f}% {r['f1']:>6.3f}")
    print(line)

# Save table
table_path = os.path.join(FIG_DIR, "fig04_comparison_table.txt")
with open(table_path, "w") as f:
    f.write("Figure 4 Sensitivity: A_max Event-Time Definition Comparison\n")
    f.write(f"Grid: {NH}x{NT}, burst ref: v3long ({np.sum(burst_ref==1)} burst)\n")
    f.write(f"v1: full run (boundary cases at 1000d)\n")
    f.write(f"v2: pre-burst (u < 0 at 32km)\n")
    f.write(f"v3: pre-burst (u < 10 m/s at 32km)\n")
    f.write(f"v4: pre-decel (5d-smooth du/dt < -1 m/s/day at 32km)\n\n")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in results:
        f.write(f"{r['variant']:<16} {r['Ac']/1e6:>10.3f} {r['accuracy']*100:>8.1f}% "
                f"{r['TP']:>5} {r['TN']:>5} {r['FP']:>4} {r['FN']:>4} "
                f"{r['precision']*100:>5.1f}% {r['recall']*100:>6.1f}% {r['f1']:>6.3f}\n")
print(f"\nSaved table: {table_path}")
print("\nDone.")
