"""
plot/plot_two_level.py
======================
Plotting functions for the two-level quasilinear model.

All functions accept output dicts from models/two_level.py.
Pass save_path (str) to save a figure instead of showing it.

Contents
--------
plot_quasilinear_solution       -- 4-panel single-run summary
plot_2level_eigenmodes          -- fixed-U eigenmode vertical structures
plot_2level_free_time_series    -- one free-mode time series
plot_2level_forced_adjustment   -- fixed-U forced adjustment (real + amplitude)
plot_event_sweep                -- hb–tau parameter sweep maps
plot_phase_sweep                -- initial free-mode phase sensitivity
plot_compare_cases_A1A2U1U2eddy -- multi-case A1, A2, U1, U2, eddy tendency
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_quasilinear_solution(result, title=None, save_path=None):
    """
    4-panel summary plot for one full-QL or nowmfi run.

    Panels: Re(A), |A|, mean wind (U1/U2/U10), eddy tendency on U1/U2.

    Parameters
    ----------
    result    : dict   Output from integrate_2level_quasilinear().
    save_path : str or None
    """
    from models.two_level import compute_u10_proxy, compute_mean_flow_tendencies
    t = result["t_days"]
    A = result["A_t"]
    U = result["U_t"]
    U10 = compute_u10_proxy(result)
    eddy_tend, _, _ = compute_mean_flow_tendencies(result)
    if title is None:
        run_type = "Full QL" if result["wave_mean_flow"] else "nowmfi"
        hb = np.real(result["hb"])
        tau = result["params"].tau_days
        title = rf"{run_type}, $h_b={hb:.1e}$, $\tau={tau}$ days"

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, np.real(A[:, 0]), label=r"Re$(A_1)$ upper")
    axes[0].plot(t, np.real(A[:, 1]), label=r"Re$(A_2)$ lower")
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set_ylabel("wave real part"); axes[0].legend(frameon=False)

    axes[1].plot(t, np.abs(A[:, 0]), label=r"$|A_1|$ upper")
    axes[1].plot(t, np.abs(A[:, 1]), label=r"$|A_2|$ lower")
    axes[1].set_ylabel("wave amplitude"); axes[1].legend(frameon=False)

    axes[2].plot(t, U[:, 0], label=r"$U_1$ upper")
    axes[2].plot(t, U[:, 1], label=r"$U_2$ lower")
    axes[2].plot(t, U10,     label=r"$U_{10}$ proxy")
    axes[2].axhline(0, linewidth=0.8)
    axes[2].set_ylabel("mean wind (m/s)"); axes[2].legend(frameon=False)

    axes[3].plot(t, eddy_tend[:, 0], label=r"eddy tendency on $U_1$")
    axes[3].plot(t, eddy_tend[:, 1], label=r"eddy tendency on $U_2$")
    axes[3].axhline(0, linewidth=0.8)
    axes[3].set_ylabel("eddy tendency\n(m/s/day)")
    axes[3].set_xlabel("time (days)"); axes[3].legend(frameon=False)

    fig.suptitle(title); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_2level_eigenmodes(lam, vec, save_path=None):
    """
    Plot the two fixed-U eigenmodes with period, decay, and mode-type annotations.

    Parameters
    ----------
    lam, vec  : eigenvalues and eigenvectors from solve_2level_free_modes_updated().
    save_path : str or None
    """
    y = np.array([0, 1])
    y_labels = [r"$A_1$ upper (~5 hPa)", r"$A_2$ lower (~30 hPa)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for j, ax in enumerate(axes):
        v = vec[:, j].copy(); v = v / np.max(np.abs(v))
        omega = np.abs(np.imag(lam[j]))
        T = np.inf if omega == 0 else 2*np.pi/omega/86400.0
        decay = -1.0/np.real(lam[j])/86400.0 if np.real(lam[j]) < 0 else np.inf
        ph = np.angle(v[1]/v[0])
        mode_type = "barotropic-like" if np.cos(ph) > 0 else "baroclinic-like"
        ax.plot(np.real(v), y, marker="o", label="real")
        ax.plot(np.imag(v), y, marker="s", linestyle="--", label="imag")
        ax.axvline(0, linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels(y_labels); ax.set_ylim(1.25, -0.25)
        ax.set_xlabel("normalized amplitude")
        ax.set_title(f"Mode {j}: {mode_type}\nperiod={T:.1f} d, decay={decay:.1f} d")
        ax.legend(frameon=False)
    axes[0].set_ylabel("vertical level")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_2level_free_time_series(lam, vec, mode_index=0,
                                  tmax_days=120, nt=1200, save_path=None):
    """
    Plot the time evolution of one free eigenmode.

    Parameters
    ----------
    lam, vec     : from solve_2level_free_modes_updated().
    mode_index   : which eigenmode to plot.
    save_path    : str or None
    """
    t_days = np.linspace(0, tmax_days, nt)
    v = vec[:, mode_index].copy(); v = v / np.max(np.abs(v))
    A_t = np.array([v * np.exp(lam[mode_index]*tt*86400.0) for tt in t_days])
    omega = np.abs(np.imag(lam[mode_index]))
    T = np.inf if omega == 0 else 2*np.pi/omega/86400.0
    decay = -1.0/np.real(lam[mode_index])/86400.0 if np.real(lam[mode_index]) < 0 else np.inf
    ph = np.angle(v[1]/v[0])
    mode_type = "barotropic-like" if np.cos(ph) > 0 else "baroclinic-like"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_days, np.real(A_t[:, 0]), label=r"Re$(A_1)$ upper, ~5 hPa")
    ax.plot(t_days, np.real(A_t[:, 1]), label=r"Re$(A_2)$ lower, ~30 hPa")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("time (days)"); ax.set_ylabel("wave amplitude")
    ax.set_title(f"Free mode {mode_index}: {mode_type}\nperiod={T:.1f} d, decay={decay:.1f} d")
    ax.legend(frameon=False); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_2level_forced_adjustment(result, title="2-level forced adjustment", save_path=None):
    """
    Plot fixed-U forced adjustment: real part and amplitude.

    Parameters
    ----------
    result    : dict   Output from solve_2level_forced_adjustment_updated().
    save_path : str or None   If given, saves both panels to one file with a suffix.
    """
    t = result["t_days"]; A_t = result["A_t"]; A_stat = result["A_stat"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, np.real(A_t[:, 0]), label=r"Re$(A_1)$ total")
    axes[0].axhline(np.real(A_stat[0]), linestyle="--", linewidth=1.5, label=r"Re$(A_{1,stat})$")
    axes[0].axhline(0, linewidth=0.7); axes[0].set_ylabel(r"$A_1$ upper"); axes[0].legend(frameon=False)
    axes[1].plot(t, np.real(A_t[:, 1]), label=r"Re$(A_2)$ total")
    axes[1].axhline(np.real(A_stat[1]), linestyle="--", linewidth=1.5, label=r"Re$(A_{2,stat})$")
    axes[1].axhline(0, linewidth=0.7); axes[1].set_ylabel(r"$A_2$ lower")
    axes[1].set_xlabel("time (days)"); axes[1].legend(frameon=False)
    fig.suptitle(title + " — real part"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_real.png"), dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, np.abs(A_t[:, 0]), label=r"$|A_1|$ upper")
    ax.plot(t, np.abs(A_t[:, 1]), label=r"$|A_2|$ lower")
    ax.axhline(np.abs(A_stat[0]), linestyle="--", linewidth=1.5, label=r"$|A_{1,stat}|$")
    ax.axhline(np.abs(A_stat[1]), linestyle=":",  linewidth=1.8, label=r"$|A_{2,stat}|$")
    ax.set_xlabel("time (days)"); ax.set_ylabel("amplitude")
    ax.set_title(title + " — amplitude"); ax.legend(frameon=False); plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png", "_amplitude.png"), dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_event_sweep(sweep_result, save_dir=None):
    """
    Plot hb–tau parameter sweep: event map, minimum wind, max drop, onset day.

    Parameters
    ----------
    sweep_result : dict   Output from sweep_hb_tau_for_events().
    save_dir     : str or None   Directory to save figures (4 separate PNGs).
    """
    import os
    hb = sweep_result["hb_values"]
    tau = sweep_result["tau_values"]
    s = sweep_result["s"]
    wm = sweep_result["wind_metric"]
    base = f"sweep_s{s}_{wm}"

    def _save(fig, name):
        if save_dir:
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    # event map
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(tau, hb, sweep_result["event_map"], shading="auto")
    ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_b$")
    ax.set_title(f"SSW-like event map, s={s}, {wm}")
    fig.colorbar(pcm, ax=ax, label="event=1 / no event=0")
    plt.tight_layout(); _save(fig, f"{base}_event_map.png")

    # reversal-only map
    if "reverse_map" in sweep_result:
        fig, ax = plt.subplots(figsize=(8, 5))
        pcm = ax.pcolormesh(tau, hb, sweep_result["reverse_map"], shading="auto")
        ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_b$")
        ax.set_title(f"Reversal-only map, s={s}, {wm}")
        fig.colorbar(pcm, ax=ax, label="reversal=1"); plt.tight_layout()
        _save(fig, f"{base}_reversal_only_map.png")

    # minimum wind
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(tau, hb, sweep_result["minU_map"], shading="auto")
    mn = sweep_result["minU_map"]
    if np.nanmin(mn) < 0.0 < np.nanmax(mn):
        ax.contour(tau, hb, mn, levels=[0.0], linewidths=2)
    ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_b$")
    ax.set_title(f"Minimum {wm}, s={s}")
    fig.colorbar(pcm, ax=ax, label=f"min {wm}"); plt.tight_layout()
    _save(fig, f"{base}_minimum_wind_map.png")

    # maximum wind drop
    if "drop_amount_map" in sweep_result:
        fig, ax = plt.subplots(figsize=(8, 5))
        pcm = ax.pcolormesh(tau, hb, sweep_result["drop_amount_map"], shading="auto")
        ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_b$")
        ax.set_title(f"Maximum wind drop in {wm}, s={s}")
        fig.colorbar(pcm, ax=ax, label=f"initial − min {wm}"); plt.tight_layout()
        _save(fig, f"{base}_maximum_wind_drop_map.png")

    # onset day
    fig, ax = plt.subplots(figsize=(8, 5))
    onset_masked = np.ma.masked_invalid(sweep_result["onset_map"])
    pcm = ax.pcolormesh(tau, hb, onset_masked, shading="auto")
    ax.set_yscale("log"); ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_b$")
    ax.set_title(f"Onset day, s={s}, {wm}")
    fig.colorbar(pcm, ax=ax, label="onset day"); plt.tight_layout()
    _save(fig, f"{base}_onset_day_map.png")


def plot_phase_sweep(phase_result, save_dir=None):
    """
    Plot initial free-mode phase sensitivity: min wind, max drop, event, onset day.

    Parameters
    ----------
    phase_result : dict   Output from sweep_initial_free_phase().
    save_dir     : str or None
    """
    import os
    phases = phase_result["phases"]
    wm = phase_result.get("wind_metric", "diagnosed wind")
    s = phase_result.get("s", "?")
    hb = phase_result.get("hb", "?")
    tau = phase_result.get("tau", "?")
    drop_thresh = phase_result.get("drop_threshold", 10.0)
    base = f"phase_sweep_s{s}_{wm}_hb{hb:.1e}_tau{tau}d".replace("+", "").replace(".", "p")

    def _save(fig, name):
        if save_dir:
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(phases, phase_result["minU"], marker="o"); ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("initial free-mode phase (rad)"); ax.set_ylabel(f"min {wm}")
    ax.set_title("Sensitivity to initial free-mode phase — minimum wind")
    plt.tight_layout(); _save(fig, f"{base}_minimum_wind.png")

    if "max_drop" in phase_result:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(phases, phase_result["max_drop"], marker="o")
        ax.axhline(drop_thresh, linestyle="--", linewidth=1.0, label=f"threshold={drop_thresh}")
        ax.set_xlabel("initial free-mode phase (rad)"); ax.set_ylabel(f"max drop in {wm}")
        ax.set_title("Wind-drop sensitivity to initial free-mode phase")
        ax.legend(frameon=False); plt.tight_layout(); _save(fig, f"{base}_maximum_wind_drop.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(phases, phase_result["event"], marker="o")
    ax.set_xlabel("initial free-mode phase (rad)"); ax.set_ylabel("event (1/0)")
    ax.set_title("Event occurrence vs initial free-mode phase")
    plt.tight_layout(); _save(fig, f"{base}_event_occurrence.png")

    onset = phase_result["onset"]; valid = np.isfinite(onset)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if np.any(valid):
        ax.plot(phases[valid], onset[valid], marker="o")
    else:
        ax.text(0.5, 0.5, "No diagnosed events for any phase",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("initial free-mode phase (rad)"); ax.set_ylabel("onset day")
    ax.set_title("Onset day vs initial free-mode phase")
    plt.tight_layout(); _save(fig, f"{base}_onset_day.png")


def plot_compare_cases_A1A2U1U2eddy(cases, wave_kind="abs", save_path=None):
    """
    Compare multiple pre-computed full-QL cases on one 6-panel figure.

    Parameters
    ----------
    cases     : list of dicts   Each from run_full_ql_case_for_comparison().
    wave_kind : "abs" or "real"
    save_path : str or None
    """
    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

    if wave_kind == "abs":
        axes[0].set_ylabel(r"$|A_1|$"); axes[1].set_ylabel(r"$|A_2|$")
        axes[0].set_title(r"Comparison: $|A_1|$, $|A_2|$, $U_1$, $U_2$, eddy tendencies")
    else:
        axes[0].set_ylabel(r"Re$(A_1)$"); axes[1].set_ylabel(r"Re$(A_2)$")
        axes[0].set_title(r"Comparison: Re$(A_1)$, Re$(A_2)$, $U_1$, $U_2$, eddy tendencies")

    for case in cases:
        label = "{} ({})".format(case.get("plot_label", ""), case.get("status", ""))
        kw = dict(label=label, color=case.get("color"), linestyle=case.get("linestyle", "-"),
                  linewidth=case.get("linewidth", 2.0))
        t = case["t"]
        y1 = case["A1_abs"] if wave_kind == "abs" else case["A1_real"]
        y2 = case["A2_abs"] if wave_kind == "abs" else case["A2_real"]
        axes[0].plot(t, y1, **kw)
        axes[1].plot(t, y2, **kw)
        axes[2].plot(t, case["U1"], **kw); axes[2].set_ylabel(r"$U_1$ (m/s)")
        axes[3].plot(t, case["U2"], **kw); axes[3].set_ylabel(r"$U_2$ (m/s)")
        axes[4].plot(t, case["eddy_tend_U1"], **kw); axes[4].set_ylabel("eddy tend.\n$U_1$ (m/s/d)")
        axes[5].plot(t, case["eddy_tend_U2"], **kw); axes[5].set_ylabel("eddy tend.\n$U_2$ (m/s/d)")

    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8); ax.grid(alpha=0.2)
    axes[5].set_xlabel("time (days)")
    axes[0].legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    else:
        plt.show()
