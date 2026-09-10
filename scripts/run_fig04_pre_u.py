"""
scripts/run_fig04_pre_u.py
==========================
Compute pre-event A_max using a configurable u-threshold at z=32km.

Usage:
  python scripts/run_fig04_pre_u.py 30
  python scripts/run_fig04_pre_u.py 10
  python scripts/run_fig04_pre_u.py 0

The argument is the u-threshold (m/s). Event time = first day u(32km) < threshold.
For non-event cases, A_max uses the full 300-day window.

Output:
  output/data/hm_model/fig04_pre_u{THRESH}.npz
  output/figures/hm_model/fig4/fig04_pre_u{THRESH}.png
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76

# ── configuration ───────────────────────────────────────────────────────────
S, DZ, IMAX, DT, ALPHA, N_DAYS = 2.0, 1000.0, 71, 360.0*15.0, False, 300
DIAG_Z_IDX = 22
SPINUP_DAYS = 50
HB_VALUES  = np.arange(11, 52, 1, dtype=float)
TAU_VALUES = np.arange(1,  62, 1, dtype=float)
N_WORKERS = 32

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR  = os.path.join(ROOT, "output", "figures", "hm_model", "fig4")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── parse CLI ───────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python scripts/run_fig04_pre_u.py <u_threshold>")
    print("  e.g. python scripts/run_fig04_pre_u.py 30")
    sys.exit(1)

U_THRESH = float(sys.argv[1])
print(f"U threshold: {U_THRESH} m/s")
print(f"Event definition: first day u(32km) < {U_THRESH} after day {SPINUP_DAYS}")


# ── worker function ─────────────────────────────────────────────────────────
def _run_one(args):
    hb, tau_d = args
    try:
        out = run_hm76(
            hb, tau_d * 86400.0,
            s=S, dz=DZ, imax=IMAX, dt=DT,
            n_days=N_DAYS, alpha_on=ALPHA,
            wave_mean_feedback=True, mean_flow_coupling=1.0,
            verbose=False,
        )
        psi_abs = np.abs(out["psi_time"][DIAG_Z_IDX, :])
        u32 = out["ud"][DIAG_Z_IDX, :]

        amax_full = float(np.nanmax(psi_abs))

        # Event time
        below = np.where(u32[SPINUP_DAYS:] < U_THRESH)[0]
        if len(below) > 0:
            t_event = int(below[0]) + SPINUP_DAYS
            amax_pre = float(np.nanmax(psi_abs[:t_event])) if t_event > 0 else 0.0
        else:
            t_event = -1
            amax_pre = amax_full

        return hb, tau_d, amax_pre, amax_full, t_event
    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        return hb, tau_d, np.nan, np.nan, -1


# ── run sweep ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]
    NH, NT = len(HB_VALUES), len(TAU_VALUES)
    total = len(tasks)

    print(f"\nGrid: {NH}x{NT} = {total} runs, {N_WORKERS} workers")
    t0 = time.time()

    amax_pre_map = np.full((NH, NT), np.nan)
    amax_full_map = np.full((NH, NT), np.nan)
    t_event_map = np.full((NH, NT), -1, dtype=int)

    hb_idx = {v: i for i, v in enumerate(HB_VALUES)}
    tau_idx = {v: j for j, v in enumerate(TAU_VALUES)}

    with mp.Pool(N_WORKERS) as pool:
        for count, (hb, tau_d, ap, af, te) in enumerate(
            pool.imap_unordered(_run_one, tasks), 1
        ):
            i, j = hb_idx[hb], tau_idx[tau_d]
            amax_pre_map[i, j] = ap
            amax_full_map[i, j] = af
            t_event_map[i, j] = te
            if count % 500 == 0 or count == total:
                elapsed = time.time() - t0
                print(f"  {count}/{total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Stats
    n_triggered = np.sum(t_event_map > 0)
    print(f"\nCases with u<{U_THRESH} event: {n_triggered} / {total}")
    if n_triggered > 0:
        te_valid = t_event_map[t_event_map > 0]
        print(f"Event times: min={te_valid.min()}, max={te_valid.max()}, "
              f"median={np.median(te_valid):.0f}")

    # Save
    tag = f"{int(U_THRESH)}" if U_THRESH == int(U_THRESH) else f"{U_THRESH:.1f}"
    npz_path = os.path.join(DATA_DIR, f"fig04_pre_u{tag}.npz")
    np.savez_compressed(npz_path,
        hb_values=HB_VALUES, tau_values=TAU_VALUES,
        amax_pre=amax_pre_map, amax_full=amax_full_map,
        t_event=t_event_map, u_thresh=np.array(U_THRESH))
    print(f"Saved: {npz_path}")

    # ── Classification ──────────────────────────────────────────────────────
    sw = np.load(os.path.join(DATA_DIR, "hb_tau_sweep_v3long.npz"))
    burst_ref = sw["burst_map"]

    valid = np.isfinite(amax_pre_map) & (burst_ref >= 0)
    af = amax_pre_map[valid]
    bf = burst_ref[valid]

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

    print(f"\nClassification (u<{U_THRESH}):")
    print(f"  Ac = {Ac:.4e}, accuracy = {best_acc*100:.1f}%")
    print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"  Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, F1={f1:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    })

    # Burst boundary
    boundary_hb, boundary_tau = [], []
    for j, tau in enumerate(TAU_VALUES):
        col = burst_ref[:, j]
        bi = np.where(col == 1)[0]
        if len(bi) > 0:
            if bi[0] > 0:
                boundary_hb.append(0.5 * (HB_VALUES[bi[0]-1] + HB_VALUES[bi[0]]))
            else:
                boundary_hb.append(HB_VALUES[0] - 0.5)
            boundary_tau.append(tau)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a)
    ax = axes[0]
    amax_plot = amax_pre_map.copy()
    amax_plot[amax_plot <= 0] = np.nan
    levs = np.linspace(np.nanmin(amax_plot), np.nanmax(amax_plot), 30)
    cf = ax.contourf(HB_VALUES, TAU_VALUES, amax_plot.T,
                     levels=levs, cmap="viridis", extend="both")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label(r"$A_\mathrm{max}$ (m$^2$ s$^{-1}$)", fontsize=14)
    ax.contour(HB_VALUES, TAU_VALUES, amax_plot.T,
               levels=[Ac], colors="red", linewidths=2.5, linestyles="-")
    if boundary_hb:
        ax.plot(boundary_hb, boundary_tau, "k--", linewidth=2.0, zorder=5)
    Ac_m = Ac / 1e6
    legend_handles = [
        Line2D([0], [0], color="red", lw=2.5, ls="-",
               label=rf"$A_c = {Ac_m:.2f} \times 10^6$ ({best_acc*100:.1f}%)"),
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="Burst boundary"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.9)
    ax.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
    ax.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
    ax.set_title(rf"(a) Pre-event $A_\mathrm{{max}}$ ($\bar{{u}} < {int(U_THRESH)}$ m/s)",
                 fontsize=14, loc="left")
    ax.set_xlim(HB_VALUES[0]-0.5, HB_VALUES[-1]+0.5)
    ax.set_ylim(TAU_VALUES[0]-0.5, TAU_VALUES[-1]+0.5)
    ax.grid(True, ls=":", alpha=0.3)

    # Panel (b)
    ax2 = axes[1]
    colors_class = {"TP": "#2ca02c", "TN": "#aec6e8", "FP": "#d62728", "FN": "#ff7f0e"}
    ms = 18
    for i in range(NH):
        for j in range(NT):
            if not valid[i, j]:
                continue
            is_burst = burst_ref[i, j] == 1
            pred_burst = amax_pre_map[i, j] >= Ac
            if is_burst and pred_burst: cat = "TP"
            elif not is_burst and not pred_burst: cat = "TN"
            elif pred_burst and not is_burst: cat = "FP"
            else: cat = "FN"
            ax2.scatter(HB_VALUES[i], TAU_VALUES[j],
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
    ax2.set_xlim(HB_VALUES[0]-0.5, HB_VALUES[-1]+0.5)
    ax2.set_ylim(TAU_VALUES[0]-0.5, TAU_VALUES[-1]+0.5)
    ax2.grid(True, ls=":", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"fig04_pre_u{tag}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")
    print("\nDone.")
