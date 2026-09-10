"""
scripts/run_fig04_sensitivity.py
================================
Figure 4 sensitivity suite: compute A_max under 4 different event-time
definitions and produce comparison figures + summary table.

Definitions:
  v1_full:    A_max = max|psi(32km, t)|  over t in [0, 300] days (no restriction)
  v2_pre_u0:  A_max = max|psi(32km, t)|  for t < t_b, where t_b = first day u(32km) < 0
              (non-burst cases: same as v1)
  v3_pre_u10: A_max = max|psi(32km, t)|  for t < t_b, where t_b = first day u(32km) < 10
              (non-burst cases: same as v1)
  v4_pre_decel: A_max = max|psi(32km, t)| for t < t_b, where t_b = first day with
              5-day-smoothed du/dt < -1.0 m/s/day at z=32km (rapid deceleration onset)
              (non-burst cases: same as v1)

Grid: hb = 11..51 (step 1), tau = 1..61 (step 1) — 41x61 = 2501 points
Burst reference: hb_tau_sweep_v3long.npz (canonical, 1000-day boundary verification)

Output:
  output/data/hm_model/fig04_sensitivity.npz
  output/figures/hm_model/fig4/fig04_v1_full.png
  output/figures/hm_model/fig4/fig04_v2_pre_u0.png
  output/figures/hm_model/fig4/fig04_v3_pre_u10.png
  output/figures/hm_model/fig4/fig04_v4_pre_decel.png
  output/figures/hm_model/fig4/fig04_comparison_table.txt

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/run_fig04_sensitivity.py
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76

# ── configuration ───────────────────────────────────────────────────────────
S     = 2.0
DZ    = 1000.0
IMAX  = 71
DT    = 360.0 * 15.0
ALPHA = False
N_DAYS = 300

DIAG_Z_IDX = 22          # z = 32 km
SPINUP_DAYS = 50          # skip for burst detection

# Deceleration criterion (v4): 5-day smoothed du/dt < DECEL_THRESH
SMOOTH_WINDOW = 5         # days
DECEL_THRESH = -1.0       # m/s/day

# Parameter grid
HB_VALUES  = np.arange(11, 52, 1, dtype=float)   # 41 values
TAU_VALUES = np.arange(1,  62, 1, dtype=float)   # 61 values

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
FIG_DIR  = os.path.join(ROOT, "output", "figures", "hm_model", "fig4")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_WORKERS = 32


# ── helper: find event time ─────────────────────────────────────────────────
def _find_event_decel(u32, smooth_win=SMOOTH_WINDOW, thresh=DECEL_THRESH,
                      spinup=SPINUP_DAYS):
    """
    Find first day when 5-day-smoothed du/dt drops below threshold.
    Returns day index or -1 if never happens.
    """
    n = len(u32)
    if n < smooth_win + spinup + 1:
        return -1
    # Central differences for du/dt (day units)
    dudt = np.zeros(n)
    dudt[1:-1] = 0.5 * (u32[2:] - u32[:-2])  # m/s per day
    dudt[0] = u32[1] - u32[0]
    dudt[-1] = u32[-1] - u32[-2]
    # Smooth with uniform window
    kernel = np.ones(smooth_win) / smooth_win
    dudt_smooth = np.convolve(dudt, kernel, mode='same')
    # Find first crossing after spinup
    below = np.where(dudt_smooth[spinup:] < thresh)[0]
    if len(below) > 0:
        return int(below[0]) + spinup
    return -1


# ── per-run function ────────────────────────────────────────────────────────
def _run_one(args):
    """Run one (hb, tau) case and return all 4 A_max variants."""
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
        n_t = len(u32)

        # v1: full-run maximum
        amax_full = float(np.nanmax(psi_abs))
        tpeak_full = int(np.nanargmax(psi_abs))

        # Event times
        # v2: first day u < 0 (after spinup)
        below_0 = np.where(u32[SPINUP_DAYS:] < 0.0)[0]
        t_event_u0 = (int(below_0[0]) + SPINUP_DAYS) if len(below_0) > 0 else -1

        # v3: first day u < 10 (after spinup)
        below_10 = np.where(u32[SPINUP_DAYS:] < 10.0)[0]
        t_event_u10 = (int(below_10[0]) + SPINUP_DAYS) if len(below_10) > 0 else -1

        # v4: first day of rapid deceleration
        t_event_decel = _find_event_decel(u32)

        # Compute pre-event A_max for each variant
        def _amax_pre(t_event):
            if t_event > 0:
                return float(np.nanmax(psi_abs[:t_event])), int(np.nanargmax(psi_abs[:t_event]))
            else:
                # No event → use full window
                return amax_full, tpeak_full

        amax_pre_u0, tpeak_pre_u0 = _amax_pre(t_event_u0)
        amax_pre_u10, tpeak_pre_u10 = _amax_pre(t_event_u10)
        amax_pre_decel, tpeak_pre_decel = _amax_pre(t_event_decel)

        # Burst flag (u < 0 after spinup)
        burst = 1 if t_event_u0 > 0 else 0
        min_u = float(np.nanmin(u32[SPINUP_DAYS:]))

        return (hb, tau_d,
                amax_full, tpeak_full,
                amax_pre_u0, tpeak_pre_u0, t_event_u0,
                amax_pre_u10, tpeak_pre_u10, t_event_u10,
                amax_pre_decel, tpeak_pre_decel, t_event_decel,
                burst, min_u)

    except Exception as e:
        print(f"  ERROR hb={hb} tau={tau_d}: {e}")
        nan7 = (np.nan, -1, np.nan, -1, -1, np.nan, -1, -1, -1, np.nan)
        return (hb, tau_d, np.nan, -1, np.nan, -1, -1, np.nan, -1, -1,
                np.nan, -1, -1, -1, np.nan)


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]
    NH, NT = len(HB_VALUES), len(TAU_VALUES)
    total = len(tasks)

    print(f"Figure 4 sensitivity sweep")
    print(f"Grid: {NH} hb x {NT} tau = {total} runs")
    print(f"Workers: {N_WORKERS}")
    print(f"Variants: v1_full, v2_pre_u0, v3_pre_u10, v4_pre_decel")
    print(f"  v4 decel criterion: {SMOOTH_WINDOW}-day smoothed du/dt < {DECEL_THRESH} m/s/day")
    print()

    t0 = time.time()

    # Allocate arrays
    amax_full_map     = np.full((NH, NT), np.nan)
    amax_pre_u0_map   = np.full((NH, NT), np.nan)
    amax_pre_u10_map  = np.full((NH, NT), np.nan)
    amax_pre_decel_map = np.full((NH, NT), np.nan)
    tpeak_full_map    = np.full((NH, NT), -1, dtype=int)
    tpeak_pre_u0_map  = np.full((NH, NT), -1, dtype=int)
    tpeak_pre_u10_map = np.full((NH, NT), -1, dtype=int)
    tpeak_pre_decel_map = np.full((NH, NT), -1, dtype=int)
    t_event_u0_map    = np.full((NH, NT), -1, dtype=int)
    t_event_u10_map   = np.full((NH, NT), -1, dtype=int)
    t_event_decel_map = np.full((NH, NT), -1, dtype=int)
    burst_map_local   = np.full((NH, NT), -1, dtype=int)
    min_u32_map       = np.full((NH, NT), np.nan)

    hb_idx  = {v: i for i, v in enumerate(HB_VALUES)}
    tau_idx = {v: j for j, v in enumerate(TAU_VALUES)}

    with mp.Pool(processes=N_WORKERS) as pool:
        for count, result in enumerate(
            pool.imap_unordered(_run_one, tasks), start=1
        ):
            (hb, tau_d,
             af, tpf,
             ap_u0, tpp_u0, te_u0,
             ap_u10, tpp_u10, te_u10,
             ap_dec, tpp_dec, te_dec,
             burst, min_u) = result

            i = hb_idx[hb]
            j = tau_idx[tau_d]
            amax_full_map[i, j] = af
            amax_pre_u0_map[i, j] = ap_u0
            amax_pre_u10_map[i, j] = ap_u10
            amax_pre_decel_map[i, j] = ap_dec
            tpeak_full_map[i, j] = tpf
            tpeak_pre_u0_map[i, j] = tpp_u0
            tpeak_pre_u10_map[i, j] = tpp_u10
            tpeak_pre_decel_map[i, j] = tpp_dec
            t_event_u0_map[i, j] = te_u0
            t_event_u10_map[i, j] = te_u10
            t_event_decel_map[i, j] = te_dec
            burst_map_local[i, j] = burst
            min_u32_map[i, j] = min_u

            if count % 100 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate if rate > 0 else 0
                print(f"  {count}/{total} done  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nAll runs finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Save data ───────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "fig04_sensitivity.npz")
    np.savez_compressed(
        out_path,
        hb_values=HB_VALUES,
        tau_values=TAU_VALUES,
        # v1
        amax_full=amax_full_map,
        tpeak_full=tpeak_full_map,
        # v2
        amax_pre_u0=amax_pre_u0_map,
        tpeak_pre_u0=tpeak_pre_u0_map,
        t_event_u0=t_event_u0_map,
        # v3
        amax_pre_u10=amax_pre_u10_map,
        tpeak_pre_u10=tpeak_pre_u10_map,
        t_event_u10=t_event_u10_map,
        # v4
        amax_pre_decel=amax_pre_decel_map,
        tpeak_pre_decel=tpeak_pre_decel_map,
        t_event_decel=t_event_decel_map,
        # metadata
        burst_map_local=burst_map_local,
        min_u32_map=min_u32_map,
        decel_smooth_window=np.array(SMOOTH_WINDOW),
        decel_threshold=np.array(DECEL_THRESH),
        diag_z_idx=np.array(DIAG_Z_IDX),
        spinup_days=np.array(SPINUP_DAYS),
    )
    print(f"\nSaved data: {out_path}")

    # ── Load canonical burst map ────────────────────────────────────────────
    burst_ref_path = os.path.join(DATA_DIR, "hb_tau_sweep_v3long.npz")
    sw = np.load(burst_ref_path)
    burst_ref = sw["burst_map"]
    print(f"Canonical burst map: {np.sum(burst_ref == 1)} burst / {burst_ref.size} total")

    # ── Classification analysis for each variant ────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })

    # Compute burst boundary
    boundary_hb, boundary_tau = [], []
    for j, tau in enumerate(TAU_VALUES):
        col = burst_ref[:, j]
        bi = np.where(col == 1)[0]
        if len(bi) > 0:
            if bi[0] > 0:
                boundary_hb.append(0.5 * (HB_VALUES[bi[0] - 1] + HB_VALUES[bi[0]]))
            else:
                boundary_hb.append(HB_VALUES[0] - 0.5)
            boundary_tau.append(tau)

    variants = [
        ("v1_full",      "Full-run maximum",              amax_full_map),
        ("v2_pre_u0",    r"Pre-burst ($\bar{u}<0$)",      amax_pre_u0_map),
        ("v3_pre_u10",   r"Pre-burst ($\bar{u}<10$ m/s)", amax_pre_u10_map),
        ("v4_pre_decel", r"Pre-deceleration (5d-smooth $d\bar{u}/dt<-1$)", amax_pre_decel_map),
    ]

    results_table = []

    for var_key, var_label, amax_map in variants:
        print(f"\n{'=' * 60}")
        print(f"  {var_key}: {var_label}")
        print(f"{'=' * 60}")

        valid = np.isfinite(amax_map) & (burst_ref >= 0)
        af = amax_map[valid]
        bf = burst_ref[valid]

        # Find optimal threshold
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

        print(f"  A_c = {Ac:.4e}")
        print(f"  Accuracy: {best_acc * 100:.1f}%")
        print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print(f"  Precision={precision * 100:.1f}%, Recall={recall * 100:.1f}%, F1={f1:.3f}")

        results_table.append({
            "variant": var_key,
            "label": var_label,
            "Ac": Ac,
            "accuracy": best_acc,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1,
        })

        # ── Plot ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel (a): contour + boundary + Ac
        ax = axes[0]
        amax_plot = amax_map.copy()
        amax_plot[amax_plot <= 0] = np.nan
        n_levs = 30
        levs = np.linspace(np.nanmin(amax_plot), np.nanmax(amax_plot), n_levs)
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
                   label=rf"$A_c = {Ac_m:.2f} \times 10^6$ ({best_acc * 100:.1f}%)"),
            Line2D([0], [0], color="k", lw=2.0, ls="--",
                   label="Burst boundary"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.9)
        ax.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
        ax.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
        ax.set_title(f"(a) {var_key}", fontsize=14, loc="left")
        ax.set_xlim(HB_VALUES[0] - 0.5, HB_VALUES[-1] + 0.5)
        ax.set_ylim(TAU_VALUES[0] - 0.5, TAU_VALUES[-1] + 0.5)
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
        ax2.set_title(f"(b) Classification (acc={best_acc * 100:.1f}%)",
                      fontsize=14, loc="left")
        ax2.set_xlim(HB_VALUES[0] - 0.5, HB_VALUES[-1] + 0.5)
        ax2.set_ylim(TAU_VALUES[0] - 0.5, TAU_VALUES[-1] + 0.5)
        ax2.grid(True, ls=":", alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, f"fig04_{var_key}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Variant':<16} {'Ac (×10⁶)':>10} {'Accuracy':>9} {'TP':>5} {'TN':>5} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>7} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(f"{r['variant']:<16} {r['Ac']/1e6:>10.3f} {r['accuracy']*100:>8.1f}% "
              f"{r['TP']:>5} {r['TN']:>5} {r['FP']:>4} {r['FN']:>4} "
              f"{r['precision']*100:>5.1f}% {r['recall']*100:>6.1f}% {r['f1']:>6.3f}")

    # Save table to file
    table_path = os.path.join(FIG_DIR, "fig04_comparison_table.txt")
    with open(table_path, "w") as f:
        f.write("Figure 4 Sensitivity: A_max Event-Time Definition Comparison\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Grid: {NH}x{NT}, N_DAYS={N_DAYS}, DIAG_Z=32km\n")
        f.write(f"Burst reference: hb_tau_sweep_v3long.npz ({np.sum(burst_ref==1)} burst)\n")
        f.write(f"v4 decel criterion: {SMOOTH_WINDOW}-day smoothed du/dt < {DECEL_THRESH} m/s/day\n")
        f.write("\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results_table:
            f.write(f"{r['variant']:<16} {r['Ac']/1e6:>10.3f} {r['accuracy']*100:>8.1f}% "
                    f"{r['TP']:>5} {r['TN']:>5} {r['FP']:>4} {r['FN']:>4} "
                    f"{r['precision']*100:>5.1f}% {r['recall']*100:>6.1f}% {r['f1']:>6.3f}\n")
    print(f"\nSaved table: {table_path}")
    print("\nDone.")
