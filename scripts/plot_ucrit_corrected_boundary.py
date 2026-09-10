"""
scripts/plot_ucrit_corrected_boundary.py
========================================
ONE updated diagnostic plot for the corrected (manuscript-U_crit) second-order
boundary. Audit styling; does NOT touch manuscript/figures/ or Fig 6.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "output", "data", "hm_model")
FIGDIR = os.path.join(ROOT, "output", "figures", "hm_model", "second_order")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"})

d = np.load(os.path.join(DATA, "second_order_boundary_corrected.npz"))
tau = d["tau_values"]
act = d["actual"]; mm0 = d["Mmax0"]
old = d["hbc_pred_old_w1000"]; corr = d["hbc_pred_corrected_w1000"]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(tau, act, "k--", lw=2.0, label="Actual transition (nonlinear)")
ax.plot(tau, mm0, color="0.5", lw=1.6, label=r"Nonlinear $M_\mathrm{max}=0$")
ax.plot(tau, corr, "-", color="#c1272d", lw=1.8,
        label=r"2nd-order pred. (manuscript $U_\mathrm{crit}$, $K_*^2$)")
ax.plot(tau, old, ":", color="#1f77b4", lw=1.4,
        label=r"2nd-order pred. (previous mixed $K^2/K_*^2$)")
ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_{b,c}$ (m)")
ax.set_title(r"Corrected second-order predicted boundary (1000-day window)")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.text(0.03, 0.06,
        f"corrected vs actual: bias +{float(d['bias']):.1f} m, "
        f"MAE {float(d['mae']):.1f} m, r={float(d['pearson']):.3f}",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGDIR, f"fig_boundary_corrected_ucrit.{ext}"),
                dpi=200 if ext == "png" else None)
plt.close(fig)
print("saved fig_boundary_corrected_ucrit.png/.pdf")
