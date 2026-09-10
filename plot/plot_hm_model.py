"""
plot/plot_hm_model.py
=====================
Plotting functions for the HM76 model.

All functions accept output dicts from models/hm76.py.
Figures are shown with plt.show() by default; pass save_path to save instead.

Contents
--------
plot_ud_with_threshold_contours  -- u(z,t) + propagation threshold contours
plot_u_zt_2x2                    -- 2×2 mean-flow comparison
plot_psi_zt_2x2                  -- 2×2 streamfunction comparison
plot_m2_contour                  -- m² contour (z,t)
plot_m2_profiles_subplots        -- m² vertical profiles, one panel per day
plot_stationary_m2_profiles_for_all_experiments
plot_phase_vs_free_reference     -- projected phase vs free-mode reference line
plot_recoeff_vs_free_marks       -- Re(projected coeff) with period marks
plot_projected_mode_diagnostics  -- complex / amplitude / phase of projection
plot_hm76_eigenmodes             -- eigenmode vertical structures
plot_hm76_eigenmode_phase        -- eigenmode vertical phase
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def plot_ud_with_threshold_contours(out_full, c_phase, title=None,
                                    vmin=-100, vmax=70, s=2.0,
                                    xlim=(150, 250), save_path=None):
    """
    u(z,t) colour-fill with critical-level threshold contours.

    Parameters
    ----------
    out_full : dict   Output from run_hm76() with betae_time.
    c_phase  : float  Transient wave phase speed (m/s).
    xlim     : tuple  Time axis limits in days.
    save_path: str or None  If given, save figure instead of showing.

    Returns
    -------
    Ucrit_stat, Ucrit_tran : ndarray
    """
    from models.hm76 import compute_Ucrit_fields
    z = out_full["z"]
    ud = out_full["ud"]
    betae_time = out_full["betae_time"]
    a = 6378000.0
    k = s / (a * np.cos(np.pi / 3.0))
    l = 3.0 / a
    ensq = 4.0e-4; h0 = 7000.0; omega_e = 7.29e-5
    f0 = 2.0 * omega_e * np.sin(np.pi / 3.0)
    eps = out_full["eps"]
    Ucrit_stat, Ucrit_tran = compute_Ucrit_fields(
        betae_time, c_phase, k=k, l=l, ensq=ensq, h0=h0, f0=f0, eps=eps)
    nt = ud.shape[1]
    t_days = np.arange(nt)
    z_km = z / 1000.0
    levels = np.linspace(vmin, vmax, 31)
    plt.figure(figsize=(10, 5))
    cf = plt.contourf(t_days, z_km, ud, cmap="RdBu_r", levels=levels, extend="both")
    plt.colorbar(cf, label="u (m/s)")
    plt.xlabel("Time (days)"); plt.ylabel("z (km)")
    cs1 = plt.contour(t_days, z_km, ud - Ucrit_stat, levels=[0],
                      colors="k", linewidths=2, linestyles="--")
    plt.clabel(cs1, fmt={0: "u=stat"}, inline=True, fontsize=9)
    cs2 = plt.contour(t_days, z_km, ud - Ucrit_tran, levels=[0],
                      colors="g", linewidths=2, linestyles="--")
    plt.clabel(cs2, fmt={0: "u=tran"}, inline=True, fontsize=9)
    plt.xlim(*xlim)
    plt.title(title or f"u(z,t) with propagation thresholds (c={c_phase:.2f} m/s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()
    return Ucrit_stat, Ucrit_tran


def _time_and_height_from_out(out):
    """Return daily time axis (days) and height axis (km) from run output."""
    nt = out["ud"].shape[1]
    t_days = np.arange(nt, dtype=float)
    z_km = out["z"] / 1000.0
    return t_days, z_km


def plot_u_zt_2x2(experiments, keys, day_start=150, day_end=250,
                  zmin=10, zmax=80, levels=21, save_path=None):
    """
    2×2 mean-flow u(z,t) comparison for four experiments.

    Parameters
    ----------
    experiments : dict  Experiment dict; each value has keys "out", "key",
                        "tau_days", "feedback_tag".
    keys        : list  Four experiment keys to plot.
    save_path   : str or None
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    panels = []; all_values = []
    for key in keys:
        exp = experiments[key]; out = exp["out"]
        t_days, z_km = _time_and_height_from_out(out)
        it = (t_days >= day_start) & (t_days <= day_end)
        iz = (z_km >= zmin) & (z_km <= zmax)
        tt = t_days[it]; zz = z_km[iz]; uu = out["ud"][np.ix_(iz, it)]
        panels.append((key, exp, tt, zz, uu)); all_values.append(uu)
    vmin = min(np.nanmin(v) for v in all_values)
    vmax_v = max(np.nanmax(v) for v in all_values)
    levs = np.linspace(vmin, vmax_v, levels)
    mappable = None
    for ax, (key, exp, tt, zz, uu) in zip(axes, panels):
        cf = ax.contourf(tt, zz, uu, levels=levs, cmap="RdBu_r", extend="both")
        mappable = cf
        title = f"{key}\ntau={exp['tau_days']:.2f} d, {exp['feedback_tag']}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (days)"); ax.set_ylabel("z (km)"); ax.grid(alpha=0.2)
    cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.92)
    cbar.set_label("u (m/s)")
    fig.suptitle("Mean flow u(z,t): nowmfi/full comparison", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_psi_zt_2x2(experiments, keys, day_start=150, day_end=250,
                    zmin=10, zmax=80, use_abs=True, levels=21, save_path=None):
    """
    2×2 streamfunction |psi|(z,t) or Re(psi)(z,t) comparison.

    Parameters
    ----------
    use_abs : bool  If True, plot |psi|; if False, plot Re(psi).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    panels = []; all_values = []
    for key in keys:
        exp = experiments[key]; out = exp["out"]
        t_days, z_km = _time_and_height_from_out(out)
        psi = out["psi_time"]
        psi_plot = np.abs(psi) if use_abs else np.real(psi)
        it = (t_days >= day_start) & (t_days <= day_end)
        iz = (z_km >= zmin) & (z_km <= zmax)
        tt = t_days[it]; zz = z_km[iz]; pp = psi_plot[np.ix_(iz, it)]
        panels.append((key, exp, tt, zz, pp)); all_values.append(pp)
    vmin = min(np.nanmin(v) for v in all_values)
    vmax_v = max(np.nanmax(v) for v in all_values)
    levs = np.linspace(vmin, vmax_v, levels)
    cmap = "viridis" if use_abs else "RdBu_r"
    mappable = None
    for ax, (key, exp, tt, zz, pp) in zip(axes, panels):
        cf = ax.contourf(tt, zz, pp, levels=levs, cmap=cmap, extend="both")
        mappable = cf
        title = f"{key}\ntau={exp['tau_days']:.2f} d, {exp['feedback_tag']}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (days)"); ax.set_ylabel("z (km)"); ax.grid(alpha=0.2)
    cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.92)
    cbar.set_label("|psi|" if use_abs else "Re(psi)")
    fig.suptitle("Streamfunction |psi|(z,t): nowmfi/full comparison", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_m2_contour(out, c=0.0, title=None, levels=31, vmax=None,
                    s=2.0, xlim=(150, 250), save_path=None):
    """
    Contour plot of m²(z,t).

    Parameters
    ----------
    out    : dict   Output from run_hm76().
    c      : float  Phase speed (m/s); c=0 for stationary wave.
    vmax   : float  Symmetric colour limit; auto if None.
    xlim   : tuple  Time axis limits in days.
    """
    from models.hm76 import compute_m2
    z_km = out["z"] / 1000.0
    nt = out["ud"].shape[1]
    time_days = np.arange(nt)
    m2 = compute_m2(out, c=c, s=s)
    m2_plot = m2 * 1e8   # units: 10⁻⁸ m⁻²
    if vmax is None:
        finite = m2_plot[np.isfinite(m2_plot)]
        vmax = np.nanpercentile(np.abs(finite), 98) if len(finite) > 0 else 1.0
    levs = np.linspace(-1, 1, 31)
    T, Z = np.meshgrid(time_days, z_km)
    plt.figure(figsize=(8, 5))
    cf = plt.contourf(T, Z, m2_plot, levels=levs, extend="both", cmap="RdBu_r")
    plt.contour(T, Z, m2_plot, levels=[0.0], colors="k", linewidths=1.5)
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$m^2$ ($10^{-8}\,\mathrm{m}^{-2}$)")
    plt.xlabel("Time (days)"); plt.ylabel("Height (km)")
    plt.xlim(*xlim)
    if title is None:
        title = rf"$m^2$ contour, $c={c:.2f}$ m s$^{{-1}}$"
    plt.title(title); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_m2_profiles_subplots(out, m2, day_start=180, day_end=200,
                              total_days=400, scale=1e8,
                              title_prefix=r"m$^2$", ncols=6,
                              figsize_per_panel=(3.2, 4.0), save_path=None):
    """
    Plot m² vertical profiles as individual subplots, one per selected day.

    Parameters
    ----------
    out          : dict    Output from run_hm76().
    m2           : ndarray Shape (nz, nt) from compute_m2().
    day_start    : float   Start of window (physical days).
    day_end      : float   End of window (physical days).
    total_days   : float   Total physical duration (= nt outputs).
    scale        : float   Scale factor for display.
    title_prefix : str     Subplot title prefix.
    ncols        : int     Number of columns.
    """
    z_km = out["z"] / 1000.0
    nt = out["ud"].shape[1]
    t_days = np.arange(nt) * (float(total_days) / nt)
    sel = np.where((t_days >= day_start) & (t_days <= day_end))[0]
    if len(sel) == 0:
        print(f"No timesteps in [{day_start}, {day_end}] d"); return
    nrows = math.ceil(len(sel) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0]*ncols, figsize_per_panel[1]*nrows),
                             sharey=True)
    axes = np.array(axes).ravel()
    xmax = np.nanpercentile(np.abs(m2[:, sel] * scale), 98) if len(sel) > 0 else 1.0
    for n, it in enumerate(sel):
        ax = axes[n]
        m2_t = m2[:, it] * scale
        ax.plot(m2_t, z_km)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(z_km[0], z_km[-1])
        ax.set_title(f"day {t_days[it]:.0f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if n % ncols == 0:
            ax.set_ylabel("z (km)", fontsize=8)
        ax.set_xlabel(f"×{scale:.0e}", fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)
    for n in range(len(sel), len(axes)):
        axes[n].set_visible(False)
    fig.suptitle(title_prefix, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_stationary_m2_profiles_for_all_experiments(
        experiments, keys, day_start=200, day_end=220,
        total_days=None, s=2.0, ncols=6, save_dir=None):
    """
    For each experiment compute stationary-wave m² and plot profiles.

    Parameters
    ----------
    save_dir : str or None  Directory to save figures; one file per experiment.
    """
    from models.hm76 import compute_m2
    import os
    for key in keys:
        exp = experiments[key]; out = exp["out"]
        total = total_days if total_days is not None else out["ud"].shape[1]
        m2_stat = compute_m2(out, c=0.0, s=s)
        title_prefix = (
            f"{key}: stationary-wave m$^2$ "
            f"(tau={exp['tau_days']:.2f} d, {exp['feedback_tag']})"
        )
        sp = os.path.join(save_dir, f"m2_stat_{key}.png") if save_dir else None
        plot_m2_profiles_subplots(
            out, m2_stat,
            day_start=day_start, day_end=day_end,
            total_days=total, title_prefix=title_prefix, ncols=ncols,
            save_path=sp,
        )


def _get_phase_series_from_projection(proj, min_amp_frac=0.05):
    """Extract unwrapped phase (in radians) from a mode projection dict."""
    t = np.asarray(proj["t_window"], dtype=float)
    coeff = np.asarray(proj["coeff_window"], dtype=np.complex128)
    amp = np.abs(coeff)
    good = amp > min_amp_frac * np.nanmax(amp) if np.nanmax(amp) > 0 else np.ones(len(amp), dtype=bool)
    return t[good], np.unwrap(np.angle(coeff[good]))


def plot_phase_vs_free_reference(mode_projections, period_table,
                                 tau_days_list=(10.91, 10.92), save_path=None):
    """
    For each tau, plot nowmfi and full unwrapped phase vs free-mode reference line.
    Phase is shown in cycles (phase / 2π) for readability.

    Parameters
    ----------
    mode_projections : dict  Keyed by experiment key, each a projection dict.
    period_table     : DataFrame  Must have columns: key, period_days, free_mode_period_days.
    """
    fig, axes = plt.subplots(len(tau_days_list), 1,
                             figsize=(9, 4.2*len(tau_days_list)), sharex=True)
    if len(tau_days_list) == 1:
        axes = [axes]
    for ax, tau_days in zip(axes, tau_days_list):
        key_full   = f"tau{str(tau_days).replace('.','p')}_full"
        key_nowmfi = f"tau{str(tau_days).replace('.','p')}_nowmfi"
        proj_now  = mode_projections[key_nowmfi]
        proj_full = mode_projections[key_full]
        row_now  = period_table[period_table["key"] == key_nowmfi].iloc[0]
        row_full = period_table[period_table["key"] == key_full].iloc[0]
        T_free = float(row_now["free_mode_period_days"])
        T_now  = float(row_now["period_days"])
        T_full = float(row_full["period_days"])
        t_now,  ph_now  = _get_phase_series_from_projection(proj_now)
        t_full, ph_full = _get_phase_series_from_projection(proj_full)
        cyc_now  = ph_now  / (2.0*np.pi) - ph_now[0]  / (2.0*np.pi)
        cyc_full = ph_full / (2.0*np.pi) - ph_full[0] / (2.0*np.pi)
        t_ref = np.linspace(min(t_now.min(), t_full.min()),
                            max(t_now.max(), t_full.max()), 300)
        cyc_ref = -(t_ref - t_ref[0]) / T_free
        ax.plot(t_now,  cyc_now,  linewidth=2.2, label=f"nowmfi, T={T_now:.1f} d")
        ax.plot(t_full, cyc_full, linewidth=2.2, label=f"full,   T={T_full:.1f} d")
        ax.plot(t_ref,  cyc_ref,  "k--", linewidth=1.8, label=f"free-mode ref, T={T_free:.1f} d")
        ax.set_ylabel("Phase (cycles)"); ax.set_title(f"tau = {tau_days:.2f} d")
        ax.grid(True, linestyle=":"); ax.legend()
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Projected free-mode phase: full vs nowmfi", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_recoeff_vs_free_marks(mode_projections, period_table,
                               tau_days_list=(10.91, 10.92), save_path=None):
    """
    Re(projected coefficient) with vertical marks every T_free days.

    Parameters
    ----------
    mode_projections : dict  Keyed by experiment key.
    period_table     : DataFrame
    """
    def _norm(y):
        y = np.asarray(y, dtype=float)
        s = np.nanmax(np.abs(y))
        return y/s if (np.isfinite(s) and s > 0) else y

    fig, axes = plt.subplots(len(tau_days_list), 1,
                             figsize=(9, 4.2*len(tau_days_list)), sharex=True)
    if len(tau_days_list) == 1:
        axes = [axes]
    for ax, tau_days in zip(axes, tau_days_list):
        key_full   = f"tau{str(tau_days).replace('.','p')}_full"
        key_nowmfi = f"tau{str(tau_days).replace('.','p')}_nowmfi"
        proj_now  = mode_projections[key_nowmfi]
        proj_full = mode_projections[key_full]
        row_now  = period_table[period_table["key"] == key_nowmfi].iloc[0]
        row_full = period_table[period_table["key"] == key_full].iloc[0]
        T_free = float(row_now["free_mode_period_days"])
        T_now  = float(row_now["period_days"])
        T_full = float(row_full["period_days"])
        t_now  = np.asarray(proj_now["t_window"],  dtype=float)
        t_full = np.asarray(proj_full["t_window"], dtype=float)
        y_now  = _norm(np.real(proj_now["coeff_window"]))
        y_full = _norm(np.real(proj_full["coeff_window"]))
        ax.plot(t_now,  y_now,  linewidth=2.2, label=f"nowmfi, T={T_now:.1f} d")
        ax.plot(t_full, y_full, linewidth=2.2, label=f"full,   T={T_full:.1f} d")
        t0 = min(t_now.min(), t_full.min()); t1 = max(t_now.max(), t_full.max())
        for tm in np.arange(t0, t1+T_free, T_free):
            ax.axvline(tm, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_ylabel("Normalized Re(coeff)")
        ax.set_title(f"tau={tau_days:.2f} d  (marks every T_free={T_free:.1f} d)")
        ax.grid(True, linestyle=":"); ax.legend()
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Projected free-mode oscillation: full vs nowmfi", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_projected_mode_diagnostics(proj, exp=None, free_period_days=None,
                                    xlim=None, save_path=None):
    """
    Three-panel diagnostic plot for a mode projection:
      1. Re and Im of complex coefficient
      2. Amplitude |coeff|
      3. Unwrapped phase

    Parameters
    ----------
    proj             : dict   Output from project_psi_time_onto_eigenmode().
    exp              : dict   Optional experiment dict for title.
    free_period_days : float  Optional free-mode period for annotation.
    xlim             : tuple  Optional time axis limits.
    """
    t = proj["t_window"]
    a = proj["coeff_window"]
    phase = np.unwrap(np.angle(a))

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    axes[0].plot(t, np.real(a), label="Re(coeff)")
    axes[0].plot(t, np.imag(a), label="Im(coeff)", linestyle="--")
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set_ylabel("complex coeff"); axes[0].grid(True, linestyle=":"); axes[0].legend()

    axes[1].plot(t, np.abs(a), label="abs(coeff)")
    axes[1].set_ylabel("amplitude"); axes[1].grid(True, linestyle=":"); axes[1].legend()

    axes[2].plot(t, phase, label="unwrapped phase")
    axes[2].set_xlabel("Time (days)"); axes[2].set_ylabel("phase (rad)")
    axes[2].grid(True, linestyle=":"); axes[2].legend()

    if free_period_days is not None and np.isfinite(free_period_days):
        for ax in axes:
            ax.text(0.02, 0.90, f"T_free={free_period_days:.2f} d",
                    transform=ax.transAxes, fontsize=9, va="top")
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    title = "Projected free-mode coefficient"
    if exp is not None:
        title = f"{exp.get('key','')}: projected free-mode coefficient"
    fig.suptitle(title); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_hm76_eigenmodes(eig_out, n_modes=4, use_abs=False, title=None, save_path=None):
    """
    Plot the first n_modes eigenmode vertical structures.

    Parameters
    ----------
    eig_out  : dict   Output from solve_hm76_eigenmodes().
    use_abs  : bool   If True, plot |A|; if False, plot Re(A).
    """
    z = eig_out["z"]; omega = eig_out["omega"]; modes = eig_out["modes"]
    order = eig_out["order"]; s = eig_out["s"]; k = eig_out["k"]
    z_km = z / 1000.0
    plt.figure(figsize=(6, 6))
    for rank, j in enumerate(order[:n_modes]):
        A = modes[:, j].copy()
        idx = np.argmax(np.abs(A)); A = A * np.exp(-1j * np.angle(A[idx]))
        wr = np.real(omega[j]); wi = np.imag(omega[j])
        T = 2.0*np.pi/np.abs(wr)/86400.0 if np.abs(wr) > 0 else np.inf
        c = wr / k
        x = np.abs(A) if use_abs else np.real(A)
        xlabel = r"$|A|$ normalized" if use_abs else r"Re$(A)$ normalized"
        label = f"rank {rank}, mode {j}: T={T:.1f} d, c={c:.1f} m/s, wi={wi:.1e}"
        plt.plot(x, z_km, label=label)
    if not use_abs:
        plt.axvline(0, color="k", linewidth=0.8)
    plt.xlabel(xlabel); plt.ylabel("Height (km)")
    plt.title(title or f"HM76 gravest free modes, s={s:g}")
    plt.grid(True, linestyle=":"); plt.legend(fontsize=8); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_hm76_eigenmode_phase(eig_out, n_modes=4, title=None, save_path=None):
    """
    Plot the vertical phase of the leading eigenmodes.

    Parameters
    ----------
    eig_out : dict   Output from solve_hm76_eigenmodes().
    """
    z = eig_out["z"]; omega = eig_out["omega"]; modes = eig_out["modes"]
    order = eig_out["order"]; s = eig_out["s"]
    z_km = z / 1000.0
    plt.figure(figsize=(6, 6))
    for rank, j in enumerate(order[:n_modes]):
        A = modes[:, j].copy()
        idx = np.argmax(np.abs(A)); A = A * np.exp(-1j * np.angle(A[idx]))
        wr = np.real(omega[j])
        T = 2.0*np.pi/np.abs(wr)/86400.0 if np.abs(wr) > 0 else np.inf
        plt.plot(np.unwrap(np.angle(A)), z_km, label=f"rank {rank}, mode {j}, T={T:.1f} d")
    plt.xlabel("Unwrapped phase (rad)"); plt.ylabel("Height (km)")
    plt.title(title or f"HM76 eigenmode phase, s={s:g}")
    plt.grid(True, linestyle=":"); plt.legend(fontsize=8); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()
