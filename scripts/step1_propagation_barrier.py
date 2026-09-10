"""
scripts/step1_propagation_barrier.py
====================================
Step 1 of the propagation-barrier analysis plan.

Diagnose the m²=0 turning level and EP-flux burst for a representative
burst case and a nearby no-burst comparison (same h_b, different tau).

Produces a multi-panel diagnostic figure for each case:
  Panel (a): ū(z,t) Hovmöller
  Panel (b): m²(z,t) Hovmöller with m²=0 contour highlighted
  Panel (c): wave amplitude |ψ|(z,t) Hovmöller (density-weighted)
  Panel (d): EP flux F_z(z,t) Hovmöller (density-weighted)
  Panel (e): time series of scalar diagnostics:
             - M(t) = min_{z∈[15,50 km]} m²
             - F_z at 32 km (density-weighted)
             - ū at 32 km
             - lowest turning level

Output:
  output/figures/hm_model/step1/step1_burst_hb30_tau10.png
  output/figures/hm_model/step1/step1_noburst_hb30_tau20.png
  output/figures/hm_model/step1/step1_comparison_timeseries.png

Usage:
  python scripts/step1_propagation_barrier.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.hm76 import (
    run_hm76, compute_m2, compute_epflux_zt,
    compute_turning_levels, compute_min_m2,
)

# ============================================================
# Configuration
# ============================================================

HB = 30.0                 # Final forcing amplitude (m)
TAU_BURST_DAYS = 10.0     # Spin-up time for burst case (days)
TAU_NOBURST_DAYS = 20.0   # Spin-up time for no-burst case (days)

S = 2.0                   # Zonal wavenumber
N_DAYS = 300              # Integration length
ALPHA_ON = False          # HM76 damping profile (False = no Newtonian cooling)
Z_DIAG_KM = 32.0         # Diagnostic altitude for time series

# Height range for the M(t) = min m² diagnostic
# The initial background wind creates a turning level at ~30 km.
# The relevant barrier region is 25–40 km: if m² becomes positive
# throughout this range, the propagation barrier is removed.
Z1_KM = 25.0
Z2_KM = 40.0

# Output directory
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "figures", "hm_model", "step1"
)

# Plotting parameters
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})


# ============================================================
# Run the two cases
# ============================================================

def run_case(hb, tau_days):
    """Run HM76 and return output dict."""
    tau_sec = tau_days * 86400.0
    print(f"Running HM76: hb={hb} m, tau={tau_days} d ({tau_sec:.0f} s), "
          f"n_days={N_DAYS}, alpha_on={ALPHA_ON}")
    out = run_hm76(hb, tau_sec, s=S, n_days=N_DAYS, alpha_on=ALPHA_ON,
                   wave_mean_feedback=True, verbose=True)
    return out


def compute_diagnostics(out):
    """Compute all Step 1 diagnostics from an HM76 output dict."""
    z = out["z"]
    z_km = z / 1000.0
    nt = out["ud"].shape[1]
    t_days = np.arange(nt, dtype=float)

    # m²(z,t)
    m2 = compute_m2(out, c=0.0, s=S)

    # EP flux F_z(z,t) — density-weighted
    Fz = compute_epflux_zt(out, density_weighted=True)

    # Wave amplitude |ψ|(z,t) — density-weighted for physical relevance
    psi_amp = np.abs(out["psi_time"])
    h0 = 7000.0
    psi_amp_rho = psi_amp * np.exp(z[:, np.newaxis] / (2.0 * h0))

    # Turning levels
    turning_levels, lowest_turning = compute_turning_levels(m2, z)

    # Scalar diagnostics
    M_t = compute_min_m2(m2, z, z1_km=Z1_KM, z2_km=Z2_KM)

    # Time series at diagnostic altitude
    iz_diag = np.argmin(np.abs(z_km - Z_DIAG_KM))
    Fz_diag = Fz[iz_diag, :]
    u_diag = out["ud"][iz_diag, :]

    return dict(
        z=z, z_km=z_km, t_days=t_days, nt=nt,
        m2=m2, Fz=Fz, psi_amp=psi_amp, psi_amp_rho=psi_amp_rho,
        turning_levels=turning_levels, lowest_turning=lowest_turning,
        M_t=M_t, Fz_diag=Fz_diag, u_diag=u_diag,
        iz_diag=iz_diag,
    )


# ============================================================
# Plotting: multi-panel Hovmöller for one case
# ============================================================

def plot_case_diagnostics(out, diag, tau_days, burst_label, save_path=None,
                          vmax_psi=None, vmax_fz=None):
    """
    5-panel diagnostic figure for one case.

    Parameters
    ----------
    vmax_psi : float or None   If given, use this as max for panel (c) wave amplitude.
    vmax_fz  : float or None   If given, use this as symmetric limit for panel (d) EP flux.
    """
    z_km = diag["z_km"]
    t = diag["t_days"]
    nt = diag["nt"]

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.35,
                           height_ratios=[1, 1, 1, 1, 1.2])

    # --- Panel (a): ū(z,t) ---
    ax_u = fig.add_subplot(gs[0])
    ud = out["ud"]
    levs_u = np.linspace(-60, 60, 31)
    cf_u = ax_u.contourf(t, z_km, ud, levels=levs_u, cmap="RdBu_r", extend="both")
    ax_u.contour(t, z_km, ud, levels=[0.0], colors="k", linewidths=1.5)
    ax_u.set_ylabel("Height (km)")
    ax_u.set_title(rf"(a)  $\bar{{u}}(z,t)$ — $h_b={HB:.0f}$ m, "
                   rf"$\tau={tau_days:.0f}$ d [{burst_label}]")
    cbar_u = fig.colorbar(cf_u, ax=ax_u, shrink=0.9, pad=0.02)
    cbar_u.set_label("m/s")
    ax_u.set_ylim(10, 80)

    # --- Panel (b): m²(z,t) with m²=0 contour ---
    ax_m2 = fig.add_subplot(gs[1])
    m2_plot = diag["m2"] * 1e8  # units: 10⁻⁸ m⁻²
    levs_m2 = np.linspace(-5, 5, 41)
    cf_m2 = ax_m2.contourf(t, z_km, m2_plot, levels=levs_m2, cmap="RdBu_r", extend="both")
    ax_m2.contour(t, z_km, m2_plot, levels=[0.0], colors="lime", linewidths=2.0)
    ax_m2.set_ylabel("Height (km)")
    ax_m2.set_title(r"(b)  $m^2(z,t)$  [green = $m^2=0$ turning level]")
    cbar_m2 = fig.colorbar(cf_m2, ax=ax_m2, shrink=0.9, pad=0.02)
    cbar_m2.set_label(r"$m^2$ ($10^{-8}$ m$^{-2}$)")
    ax_m2.set_ylim(10, 80)

    # --- Panel (c): wave amplitude |ψ|*√ρ(z,t) ---
    ax_psi = fig.add_subplot(gs[2])
    psi_rho = diag["psi_amp_rho"]
    if vmax_psi is None:
        vmax_psi = np.nanpercentile(psi_rho, 99)
    levs_psi = np.linspace(0, vmax_psi, 31)
    cf_psi = ax_psi.contourf(t, z_km, psi_rho, levels=levs_psi,
                             cmap="hot_r", extend="max")
    ax_psi.set_ylabel("Height (km)")
    ax_psi.set_title(r"(c)  Wave amplitude $|\psi| \cdot e^{z/2H}$ (density-weighted)")
    cbar_psi = fig.colorbar(cf_psi, ax=ax_psi, shrink=0.9, pad=0.02)
    cbar_psi.set_label(r"m$^2$/s")
    ax_psi.set_ylim(10, 80)

    # --- Panel (d): EP flux F_z(z,t) density-weighted ---
    ax_fz = fig.add_subplot(gs[3])
    Fz = diag["Fz"]
    if vmax_fz is None:
        vmax_fz = np.nanpercentile(np.abs(Fz[5:-5, :]), 98)
    levs_fz = np.linspace(-vmax_fz, vmax_fz, 31)
    cf_fz = ax_fz.contourf(t, z_km, Fz, levels=levs_fz, cmap="RdBu_r", extend="both")
    ax_fz.set_ylabel("Height (km)")
    ax_fz.set_title(r"(d)  EP flux $F_z(z,t) \cdot e^{z/H}$ (density-weighted, upward positive)")
    cbar_fz = fig.colorbar(cf_fz, ax=ax_fz, shrink=0.9, pad=0.02)
    ax_fz.set_ylim(10, 80)

    # --- Panel (e): scalar time series showing M(t) vs ū ---
    # Both y-axes aligned at zero so one horizontal line shows both crossings
    # Full time range
    ax_ts = fig.add_subplot(gs[4])

    color_M = "C0"
    color_u = "C1"

    # M(t) on left axis — range [-5, 2] → zero at fraction 5/7 from bottom
    M_lo, M_hi = -5.0, 2.0
    ax_ts.plot(t, diag["M_t"] * 1e8, color=color_M, lw=2,
               label=r"$M(t) = \min_{z} m^2$ ($10^{-8}$ m$^{-2}$)")
    ax_ts.set_ylim(M_lo, M_hi)
    ax_ts.set_ylabel(r"$M(t)$ ($10^{-8}$ m$^{-2}$)", color=color_M, fontsize=12)
    ax_ts.tick_params(axis="y", labelcolor=color_M)
    ax_ts.set_xlabel("Time (days)")

    # ū on right axis — align zero with M(t) zero
    zero_frac = (0 - M_lo) / (M_hi - M_lo)
    u_hi = 40.0
    u_lo = -u_hi * zero_frac / (1.0 - zero_frac)
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(t, diag["u_diag"], color=color_u, lw=2, ls="--",
                label=rf"$\bar{{u}}$ at {Z_DIAG_KM:.0f} km")
    ax_ts2.set_ylim(u_lo, u_hi)
    ax_ts2.set_ylabel(r"$\bar{u}$ (m/s)", color=color_u, fontsize=12)
    ax_ts2.tick_params(axis="y", labelcolor=color_u)

    # Single zero line — applies to both
    ax_ts.axhline(0, color="k", lw=1.2, ls="--", zorder=5)

    ax_ts.set_title(r"(e)  $M(t)$ and $\bar{u}$ — zeros aligned, 25–40 km")

    # Combined legend
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ============================================================
# Plotting: comparison time series (both cases side by side)
# ============================================================

def plot_comparison_timeseries(diag_burst, diag_noburst, save_path=None):
    """
    4-panel comparison time-series figure with vertical line at M(t)=0 crossing.
    """
    t_b = diag_burst["t_days"]
    t_nb = diag_noburst["t_days"]

    # Find the time when M(t) first crosses zero in the burst case
    M_burst = diag_burst["M_t"] * 1e8
    t_M_cross = None
    for i in range(1, len(M_burst)):
        if M_burst[i - 1] < 0 and M_burst[i] >= 0:
            t_M_cross = t_b[i]
            break

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # (a) M(t) = min m² over 25–40 km
    ax = axes[0]
    ax.plot(t_b, diag_burst["M_t"] * 1e8, "C0", lw=2,
            label=rf"Burst ($\tau={TAU_BURST_DAYS:.0f}$ d)")
    ax.plot(t_nb, diag_noburst["M_t"] * 1e8, "C1", lw=2,
            label=rf"No burst ($\tau={TAU_NOBURST_DAYS:.0f}$ d)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(-5.0, 1.0)
    ax.set_ylabel(r"$M(t) = \min_z m^2$" + "\n" + r"($10^{-8}$ m$^{-2}$, 25–40 km)")
    ax.set_title(rf"(a)  Propagation barrier diagnostic — $h_b = {HB:.0f}$ m")
    ax.legend(loc="lower left")
    ax.grid(True, ls=":", alpha=0.4)

    # (b) EP flux at diagnostic altitude
    ax = axes[1]
    ax.plot(t_b, diag_burst["Fz_diag"], "C0", lw=1.5,
            label=rf"Burst ($\tau={TAU_BURST_DAYS:.0f}$ d)")
    ax.plot(t_nb, diag_noburst["Fz_diag"], "C1", lw=1.5,
            label=rf"No burst ($\tau={TAU_NOBURST_DAYS:.0f}$ d)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(rf"$F_z$ at {Z_DIAG_KM:.0f} km" + "\n(density-weighted)")
    ax.set_title(r"(b)  Vertical EP flux at diagnostic altitude")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.4)

    # (c) Mean wind at diagnostic altitude
    ax = axes[2]
    ax.plot(t_b, diag_burst["u_diag"], "C0", lw=2,
            label=rf"Burst ($\tau={TAU_BURST_DAYS:.0f}$ d)")
    ax.plot(t_nb, diag_noburst["u_diag"], "C1", lw=2,
            label=rf"No burst ($\tau={TAU_NOBURST_DAYS:.0f}$ d)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(rf"$\bar{{u}}$ at {Z_DIAG_KM:.0f} km (m/s)")
    ax.set_title(r"(c)  Mean zonal wind at diagnostic altitude")
    ax.legend(loc="lower left")
    ax.grid(True, ls=":", alpha=0.4)

    # (d) Lowest turning level
    ax = axes[3]
    lt_b = diag_burst["lowest_turning"] / 1000.0
    lt_nb = diag_noburst["lowest_turning"] / 1000.0
    valid_b = np.isfinite(lt_b)
    valid_nb = np.isfinite(lt_nb)
    if np.any(valid_b):
        ax.plot(t_b[valid_b], lt_b[valid_b], "C0", lw=1.5, marker=".", ms=2,
                label=rf"Burst ($\tau={TAU_BURST_DAYS:.0f}$ d)")
    if np.any(valid_nb):
        ax.plot(t_nb[valid_nb], lt_nb[valid_nb], "C1", lw=1.5, marker=".", ms=2,
                label=rf"No burst ($\tau={TAU_NOBURST_DAYS:.0f}$ d)")
    ax.set_ylabel("Lowest turning level (km)")
    ax.set_xlabel("Time (days)")
    ax.set_title(r"(d)  Lowest $m^2=0$ turning level altitude")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_ylim(10, 85)

    # Add vertical line at M(t)=0 crossing to ALL panels
    if t_M_cross is not None:
        for ax in axes:
            ax.axvline(t_M_cross, color="gray", lw=1.5, ls="--", alpha=0.7)
        # Label only on the top panel
        axes[0].text(t_M_cross + 2, 0.6, rf"$M=0$: day {t_M_cross:.0f}",
                     color="gray", fontsize=10, ha="left", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Step 1: Propagation barrier diagnosis")
    print(f"  Burst case:    hb={HB}, tau={TAU_BURST_DAYS} d")
    print(f"  No-burst case: hb={HB}, tau={TAU_NOBURST_DAYS} d")
    print("=" * 70)

    # --- Run burst case ---
    print("\n--- BURST CASE ---")
    out_burst = run_case(HB, TAU_BURST_DAYS)
    diag_burst = compute_diagnostics(out_burst)
    print(f"  Min u at {Z_DIAG_KM} km: {diag_burst['u_diag'].min():.1f} m/s")
    print(f"  Min M(t): {diag_burst['M_t'].min() * 1e8:.2f} (10⁻⁸ m⁻²)")

    # --- Run no-burst case ---
    print("\n--- NO-BURST CASE ---")
    out_noburst = run_case(HB, TAU_NOBURST_DAYS)
    diag_noburst = compute_diagnostics(out_noburst)
    print(f"  Min u at {Z_DIAG_KM} km: {diag_noburst['u_diag'].min():.1f} m/s")
    print(f"  Min M(t): {diag_noburst['M_t'].min() * 1e8:.2f} (10⁻⁸ m⁻²)")

    # --- Plot individual case diagnostics ---
    # Use burst case color ranges for both panels (c) and (d)
    vmax_psi_shared = float(np.nanpercentile(diag_burst["psi_amp_rho"], 99))
    vmax_fz_shared = float(np.nanpercentile(
        np.abs(diag_burst["Fz"][5:-5, :]), 98))

    print("\n--- Plotting burst case ---")
    plot_case_diagnostics(
        out_burst, diag_burst, TAU_BURST_DAYS, "BURST",
        save_path=os.path.join(OUT_DIR, "step1_burst_hb30_tau10.png"),
        vmax_psi=vmax_psi_shared, vmax_fz=vmax_fz_shared,
    )

    print("--- Plotting no-burst case ---")
    plot_case_diagnostics(
        out_noburst, diag_noburst, TAU_NOBURST_DAYS, "NO BURST",
        save_path=os.path.join(OUT_DIR, "step1_noburst_hb30_tau20.png"),
        vmax_psi=vmax_psi_shared, vmax_fz=vmax_fz_shared,
    )

    # --- Plot comparison ---
    print("--- Plotting comparison ---")
    plot_comparison_timeseries(
        diag_burst, diag_noburst,
        save_path=os.path.join(OUT_DIR, "step1_comparison_timeseries.png"),
    )

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Burst case (hb={HB}, tau={TAU_BURST_DAYS} d):")
    print(f"  Min ū at {Z_DIAG_KM} km: {diag_burst['u_diag'].min():.1f} m/s "
          f"(day {diag_burst['t_days'][np.argmin(diag_burst['u_diag'])]:.0f})")
    print(f"  Min M(t): {diag_burst['M_t'].min() * 1e8:.3f} × 10⁻⁸ m⁻² "
          f"(day {diag_burst['t_days'][np.nanargmin(diag_burst['M_t'])]:.0f})")

    # Check if M(t) crosses zero before burst
    M_burst = diag_burst["M_t"]
    zero_crossings = np.where(M_burst[:-1] * M_burst[1:] < 0)[0]
    if len(zero_crossings) > 0:
        first_cross_day = diag_burst["t_days"][zero_crossings[0]]
        print(f"  M(t) first crosses zero at day {first_cross_day:.0f}")
    else:
        if np.any(M_burst < 0):
            first_neg = np.where(M_burst < 0)[0][0]
            print(f"  M(t) first goes negative at day "
                  f"{diag_burst['t_days'][first_neg]:.0f}")
        else:
            print("  M(t) never goes negative!")

    # Burst onset (u < 0 at diagnostic level)
    u_burst = diag_burst["u_diag"]
    reversal_idx = np.where(u_burst < 0)[0]
    if len(reversal_idx) > 0:
        onset_day = diag_burst["t_days"][reversal_idx[0]]
        print(f"  Wind reversal (u<0) onset: day {onset_day:.0f}")
    else:
        print("  No wind reversal (u<0) within integration.")

    print(f"\nNo-burst case (hb={HB}, tau={TAU_NOBURST_DAYS} d):")
    print(f"  Min ū at {Z_DIAG_KM} km: {diag_noburst['u_diag'].min():.1f} m/s")
    print(f"  Min M(t): {diag_noburst['M_t'].min() * 1e8:.3f} × 10⁻⁸ m⁻²")
    M_nb = diag_noburst["M_t"]
    if np.any(M_nb < 0):
        first_neg = np.where(M_nb < 0)[0][0]
        print(f"  M(t) first goes negative at day "
              f"{diag_noburst['t_days'][first_neg]:.0f}")
    else:
        print("  M(t) never goes negative → propagation barrier persists.")

    print(f"\nOutput saved to: {OUT_DIR}/")
    print("Done.")
