"""
plot/plot_fp_timeseries.py
==========================
Plot time series of a False Positive case from the u-threshold sensitivity test.

Shows |psi| and u at z=32km, with markers for:
  - u-threshold event time (vertical green dashed)
  - peak |psi| before event (purple dot)
  - u=0 line (if applicable)

Usage:
  python plot/plot_fp_timeseries.py [hb] [tau] [u_thresh]
  python plot/plot_fp_timeseries.py 19 1 30

If no arguments given, uses a default representative FP case.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import run_hm76

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})

FIG_DIR = os.path.join(ROOT, "output", "figures", "hm_model", "fig4")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_fp_timeseries(hb, tau_days, u_thresh=30.0, save=True, dpi=200):
    """
    Run a single (hb, tau) case and plot time series showing why it's an FP.

    Parameters
    ----------
    hb : float
        Forcing amplitude (m)
    tau_days : float
        Spin-up time (days)
    u_thresh : float
        U-threshold used for event definition (m/s)
    save : bool
        If True, save to output/figures/hm_model/fig4/
    dpi : int
        Figure resolution
    """
    # Run model
    out = run_hm76(
        hb, tau_days * 86400.0,
        s=2.0, dz=1000.0, imax=71, dt=360.0*15.0,
        n_days=300, alpha_on=False,
        wave_mean_feedback=True, mean_flow_coupling=1.0,
        verbose=False,
    )

    DIAG_Z_IDX = 22
    SPINUP_DAYS = 50

    psi_abs = np.abs(out["psi_time"][DIAG_Z_IDX, :])
    u32 = out["ud"][DIAG_Z_IDX, :]
    t_d = np.arange(len(u32), dtype=float)

    # Event time: u < u_thresh after spinup
    below = np.where(u32[SPINUP_DAYS:] < u_thresh)[0]
    t_event = (int(below[0]) + SPINUP_DAYS) if len(below) > 0 else -1

    # Peak amplitude before event
    if t_event > 0:
        amax_pre = np.nanmax(psi_abs[:t_event])
        tpeak_pre = int(np.nanargmax(psi_abs[:t_event]))
    else:
        amax_pre = np.nanmax(psi_abs)
        tpeak_pre = int(np.nanargmax(psi_abs))

    # Full peak
    amax_full = np.nanmax(psi_abs)
    tpeak_full = int(np.nanargmax(psi_abs))

    # Burst time (u < 0)
    below_0 = np.where(u32[SPINUP_DAYS:] < 0.0)[0]
    t_burst = (int(below_0[0]) + SPINUP_DAYS) if len(below_0) > 0 else -1

    # Check if it's actually a no-burst case
    is_burst = t_burst > 0
    min_u = float(np.nanmin(u32[SPINUP_DAYS:]))

    print(f"Case: hb={hb}, tau={tau_days} d")
    print(f"  u_thresh={u_thresh}, t_event={t_event}")
    print(f"  A_max_pre={amax_pre:.4e} (day {tpeak_pre})")
    print(f"  A_max_full={amax_full:.4e} (day {tpeak_full})")
    print(f"  min_u={min_u:.1f}, burst={'YES day '+str(t_burst) if is_burst else 'NO'}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Panel 1: |psi| at 32 km
    ax1.plot(t_d, psi_abs, "b-", lw=1.8, label=r"$|\psi'|$ at 32 km")

    # Mark peak amplitude (before event)
    ax1.plot(tpeak_pre, amax_pre, "o", color="#9467bd", ms=10, zorder=5,
             label=f"Peak before event (day {tpeak_pre})")

    # Mark full peak if different
    if tpeak_full != tpeak_pre:
        ax1.plot(tpeak_full, amax_full, "s", color="#d62728", ms=8, zorder=5,
                 label=f"Full-run peak (day {tpeak_full})")

    # Event time (u < threshold)
    if t_event > 0:
        ax1.axvline(t_event, color="#2ca02c", ls="--", lw=2, alpha=0.8,
                    label=rf"$\bar{{u}} < {int(u_thresh)}$ (day {t_event})")

    # Burst time
    if t_burst > 0:
        ax1.axvline(t_burst, color="#d62728", ls="-.", lw=2, alpha=0.8,
                    label=rf"$\bar{{u}} < 0$ (day {t_burst})")

    ax1.set_ylabel(r"$|\psi'|$ (m$^2$ s$^{-1}$)", fontsize=15)
    ax1.set_title(rf"FP case: $h_b = {hb}$ m, $\tau = {tau_days}$ d "
                  rf"(event: $\bar{{u}} < {int(u_thresh)}$ m/s)",
                  fontsize=14, loc="left")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, ls=":", alpha=0.3)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Panel 2: u at 32 km
    ax2.plot(t_d, u32, "k-", lw=1.8, label=r"$\bar{u}$ at 32 km")
    ax2.axhline(u_thresh, color="#2ca02c", ls="--", lw=1.5, alpha=0.7,
                label=rf"$\bar{{u}} = {int(u_thresh)}$ m/s")
    ax2.axhline(0, color="#d62728", ls="-.", lw=1.5, alpha=0.7,
                label=r"$\bar{u} = 0$")

    if t_event > 0:
        ax2.axvline(t_event, color="#2ca02c", ls="--", lw=2, alpha=0.8)
    if t_burst > 0:
        ax2.axvline(t_burst, color="#d62728", ls="-.", lw=2, alpha=0.8)

    ax2.set_xlabel("Time (days)", fontsize=15)
    ax2.set_ylabel(r"$\bar{u}$ (m s$^{-1}$)", fontsize=15)
    ax2.set_xlim(0, 300)
    ax2.legend(fontsize=11, loc="lower left")
    ax2.grid(True, ls=":", alpha=0.3)

    # Annotate: this is NOT a burst case
    if not is_burst:
        ax2.text(0.98, 0.05, "NO BURST (FP case)",
                 transform=ax2.transAxes, fontsize=13, color="#d62728",
                 ha="right", va="bottom", weight="bold",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="#d62728"))

    plt.tight_layout()

    if save:
        tag = f"hb{int(hb)}_tau{int(tau_days)}_u{int(u_thresh)}"
        save_path = os.path.join(FIG_DIR, f"fp_timeseries_{tag}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
        return save_path
    else:
        plt.close(fig)
        return None


def plot_fn_timeseries(hb, tau_days, n_days=1000, Ac=5.76e6, save=True, dpi=200):
    """
    Plot time series of any case, auto-detecting TP/TN/FN classification.

    Runs the model for n_days and shows:
      - |psi| with Ac threshold line and 300-day boundary
      - u with burst onset marker

    Parameters
    ----------
    hb : float
        Forcing amplitude (m)
    tau_days : float
        Spin-up time (days)
    n_days : int
        Integration length (needs >300 for late-burst cases)
    Ac : float
        Critical amplitude threshold for reference
    save : bool
        If True, save figure
    dpi : int
        Figure resolution
    """
    out = run_hm76(
        hb, tau_days * 86400.0,
        s=2.0, dz=1000.0, imax=71, dt=360.0*15.0,
        n_days=n_days, alpha_on=False,
        wave_mean_feedback=True, mean_flow_coupling=1.0,
        verbose=False,
    )

    DIAG_Z_IDX = 22
    SPINUP_DAYS = 50

    psi_abs = np.abs(out["psi_time"][DIAG_Z_IDX, :])
    u32 = out["ud"][DIAG_Z_IDX, :]
    t_d = np.arange(len(u32), dtype=float)

    # Burst onset (u < 0)
    below_0 = np.where(u32[SPINUP_DAYS:] < 0.0)[0]
    t_burst = (int(below_0[0]) + SPINUP_DAYS) if len(below_0) > 0 else -1

    # Peak amplitude (full run)
    tpeak = int(np.nanargmax(psi_abs))
    amax = float(np.nanmax(psi_abs))

    # Peak within 300 days
    psi_300 = psi_abs[:min(300, len(psi_abs))]
    tpeak_300 = int(np.nanargmax(psi_300))
    amax_300 = float(np.nanmax(psi_300))

    # Auto-detect classification
    is_burst = t_burst > 0
    exceeds_Ac = amax_300 >= Ac
    if is_burst and exceeds_Ac:
        case_label = "TP"
        case_color = "#2ca02c"
        case_desc = f"BURST (day {t_burst}), $A_{{\\max}} > A_c$"
    elif not is_burst and not exceeds_Ac:
        case_label = "TN"
        case_color = "#1f77b4"
        case_desc = r"NO BURST, $A_{\max} < A_c$"
    elif is_burst and not exceeds_Ac:
        case_label = "FN"
        case_color = "#ff7f0e"
        if t_burst > 300:
            case_desc = f"LATE BURST (day {t_burst}, after 300d)"
        else:
            case_desc = f"BURST (day {t_burst}), but $A_{{\\max}}(300d) < A_c$"
    else:
        case_label = "FP"
        case_color = "#d62728"
        case_desc = r"NO BURST, but $A_{\max} > A_c$"

    print(f"Case: hb={hb}, tau={tau_days} d → {case_label}")
    print(f"  t_burst (u<0) = day {t_burst}")
    print(f"  A_max ({n_days}d) = {amax:.4e} at day {tpeak}")
    print(f"  A_max (300d)  = {amax_300:.4e} at day {tpeak_300}")
    print(f"  Ac = {Ac:.4e}")

    # ── Plot ────────────────────────────────────────────────────────────────
    t_max_plot = min(500, n_days)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    mask = t_d <= t_max_plot

    # Panel 1: |psi|
    ax1.plot(t_d[mask], psi_abs[mask], "b-", lw=1.8, label=r"$|\psi'|$ at 32 km")
    ax1.axhline(Ac, color="red", ls="--", lw=1.5, alpha=0.8,
                label=rf"$A_c = {Ac/1e6:.2f} \times 10^6$")
    ax1.axvline(300, color="grey", ls=":", lw=2, alpha=0.7, label="300-day window")

    # Mark peak within 300d
    ax1.plot(tpeak_300, amax_300, "o", color="#9467bd", ms=10, zorder=5,
             label=f"Peak in 300d (day {tpeak_300})")

    # Mark full peak (if within plot range and different)
    if tpeak <= t_max_plot and tpeak != tpeak_300:
        ax1.plot(tpeak, amax, "s", color="#d62728", ms=8, zorder=5,
                 label=f"Full peak (day {tpeak})")

    # Mark burst onset
    if t_burst > 0 and t_burst <= t_max_plot:
        ax1.axvline(t_burst, color="#ff7f0e", ls="-.", lw=2, alpha=0.8,
                    label=f"Burst onset (day {t_burst})")

    ax1.set_ylabel(r"$|\psi'|$ (m$^2$ s$^{-1}$)", fontsize=15)
    ax1.set_title(rf"{case_label} case: $h_b = {int(hb)}$ m, $\tau = {int(tau_days)}$ d",
                  fontsize=14, loc="left")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, ls=":", alpha=0.3)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Panel 2: u
    ax2.plot(t_d[mask], u32[mask], "k-", lw=1.8, label=r"$\bar{u}$ at 32 km")
    ax2.axhline(0, color="#d62728", ls="-.", lw=1.5, alpha=0.7, label=r"$\bar{u} = 0$")
    ax2.axvline(300, color="grey", ls=":", lw=2, alpha=0.7)

    if t_burst > 0 and t_burst <= t_max_plot:
        ax2.axvline(t_burst, color="#ff7f0e", ls="-.", lw=2, alpha=0.8,
                    label=f"Burst onset (day {t_burst})")

    ax2.set_xlabel("Time (days)", fontsize=15)
    ax2.set_ylabel(r"$\bar{u}$ (m s$^{-1}$)", fontsize=15)
    ax2.set_xlim(0, t_max_plot)
    ax2.legend(fontsize=11, loc="lower left")
    ax2.grid(True, ls=":", alpha=0.3)

    # Annotate with case classification
    ax2.text(0.98, 0.95, f"{case_label}: {case_desc}",
             transform=ax2.transAxes, fontsize=13, color=case_color,
             ha="right", va="top", weight="bold",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor=case_color))

    plt.tight_layout()

    if save:
        tag = f"hb{int(hb)}_tau{int(tau_days)}"
        save_path = os.path.join(FIG_DIR, f"case_timeseries_{tag}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
        return save_path
    else:
        plt.close(fig)
        return None


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot FP or case time series")
    parser.add_argument("mode", choices=["fp", "case"], help="fp (u-thresh FP) or case (auto-detect TP/TN/FN)")
    parser.add_argument("hb", type=float, help="Forcing amplitude (m)")
    parser.add_argument("tau", type=float, help="Spin-up time (days)")
    parser.add_argument("--u_thresh", type=float, default=30.0,
                        help="U threshold for FP mode (default: 30)")
    parser.add_argument("--n_days", type=int, default=1000,
                        help="Integration days for case mode (default: 1000)")
    parser.add_argument("--Ac", type=float, default=5.76e6,
                        help="Critical amplitude for reference line")
    args = parser.parse_args()

    if args.mode == "fp":
        plot_fp_timeseries(args.hb, args.tau, u_thresh=args.u_thresh)
    else:
        plot_fn_timeseries(args.hb, args.tau, n_days=args.n_days, Ac=args.Ac)
