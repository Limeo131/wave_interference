"""
scripts/step2_free_modes.py
===========================
Step 2: Compute linear free modes for a prescribed background wind.

Takes the early-time ū(z) profile from the burst case (before the transition),
freezes it as the background, and solves the linear eigenvalue problem for
free Rossby-wave modes with ψ=0 at the lower boundary.

Compares the free-mode periods with the observed wave-activity vacillation period
from the HM76 simulation.

Output:
  output/figures/hm_model/step2/step2_free_modes_structure.png
  output/figures/hm_model/step2/step2_modes_vs_m2.png
  output/figures/hm_model/step2/step2_period_comparison.png

Usage:
  python scripts/step2_free_modes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from models.hm76 import (
    run_hm76, compute_m2, solve_hm76_eigenmodes,
    get_eigenmode_info, eigenmode_period_table,
    hm76_constants, compute_betae_for_eigen,
)

# ============================================================
# Configuration
# ============================================================

HB = 30.0                 # Final forcing amplitude (m)
TAU_BURST_DAYS = 10.0     # Spin-up time (burst case)
S = 2.0                   # Zonal wavenumber
N_DAYS = 300              # Integration length
ALPHA_ON = False          # HM76 damping (same as Step 1)

# Day at which to extract the background wind for the eigenvalue problem
# Use day 0 = initial background wind, which is the SAME wind frozen in the no-WMFI run
T_PROFILE_DAY = 0

# Number of leading eigenmodes to analyze
N_MODES = 6

# Output directory
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "figures", "hm_model", "step2"
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Step 2: Compute linear free modes for prescribed background wind")
    print(f"  Case: hb={HB}, tau={TAU_BURST_DAYS} d (burst)")
    print(f"  Background wind extracted at day {T_PROFILE_DAY}")
    print(f"  alpha_on={ALPHA_ON}")
    print("=" * 70)

    # --- Run the burst case to get the early-time wind profile ---
    print("\nRunning HM76 burst case...")
    tau_sec = TAU_BURST_DAYS * 86400.0
    out = run_hm76(HB, tau_sec, s=S, n_days=N_DAYS, alpha_on=ALPHA_ON,
                   wave_mean_feedback=True, verbose=True)

    z = out["z"]
    z_km = z / 1000.0
    ud = out["ud"]

    # Extract ū(z) at the chosen day
    u_profile = ud[:, T_PROFILE_DAY].copy()
    print(f"\nBackground wind profile at day {T_PROFILE_DAY}:")
    print(f"  u(10 km) = {u_profile[0]:.1f} m/s")
    print(f"  u(30 km) = {u_profile[20]:.1f} m/s")
    print(f"  u(50 km) = {u_profile[40]:.1f} m/s")
    print(f"  u(70 km) = {u_profile[60]:.1f} m/s")

    # Also get the initial (day 0) profile for comparison
    u_initial = ud[:, 0].copy()

    # --- Compute m²(z) for the chosen background wind ---
    # Need to build a mock 'out' dict with the frozen wind to compute m²
    # Actually compute_m2 uses betae_time and ud from the full run,
    # so let's compute m² directly for the frozen profile
    const = hm76_constants(s=S, dz=1000.0)
    k = const["k"]; l = const["l"]; eps = const["eps"]
    f0 = const["f0"]; ensq = const["ensq"]; h0 = const["h0"]
    LD_inv2 = const["LD_inv2"]
    K2 = k**2 + l**2 + LD_inv2

    betae_profile = np.real(compute_betae_for_eigen(z, u_profile, const))
    m2_profile = (ensq / f0**2) * (betae_profile / (eps * u_profile) - K2)

    # Find turning level from this profile
    turning_heights = []
    for i in range(len(z) - 1):
        if m2_profile[i] * m2_profile[i + 1] < 0:
            z_cross = z[i] + (z[i + 1] - z[i]) * (-m2_profile[i]) / (m2_profile[i + 1] - m2_profile[i])
            turning_heights.append(z_cross / 1000.0)

    print(f"\nm²(z) turning levels at day {T_PROFILE_DAY}: {[f'{h:.1f}' for h in turning_heights]} km")

    # --- Solve the eigenvalue problem ---
    print("\nSolving eigenvalue problem...")
    eig_out = solve_hm76_eigenmodes(s=S, dz=1000.0, imax=71,
                                     alpha_on=ALPHA_ON, u_bg=u_profile,
                                     n_print=N_MODES)

    # Extract leading modes info
    order = eig_out["order"]
    omega = eig_out["omega"]
    modes = eig_out["modes"]

    print(f"\nLeading {N_MODES} free modes:")
    print("-" * 80)
    mode_info = []
    for rank in range(N_MODES):
        j = order[rank]
        wr = np.real(omega[j])
        wi = np.imag(omega[j])
        T_days = 2.0 * np.pi / np.abs(wr) / 86400.0 if np.abs(wr) > 0 else np.inf
        c_phase = wr / k
        decay_days = -1.0 / wi / 86400.0 if wi < 0 else np.inf
        nodes = eig_out["node_count"][j]
        centroid = eig_out["centroid_km"][j]
        mode_info.append(dict(rank=rank, j=j, T_days=T_days, c_phase=c_phase,
                              decay_days=decay_days, nodes=nodes, centroid_km=centroid,
                              wr=wr, wi=wi))
        print(f"  Mode {rank}: T = {T_days:.1f} d, c = {c_phase:.2f} m/s, "
              f"decay = {decay_days:.1f} d, nodes = {nodes}, centroid = {centroid:.1f} km")

    # --- Estimate the observed vacillation period from the simulation ---
    # Use the no-WMFI case for clean comparison (frozen mean flow = linear)
    print("\nRunning no-WMFI case for vacillation period comparison...")
    out_nowmfi = run_hm76(HB, tau_sec, s=S, n_days=N_DAYS, alpha_on=ALPHA_ON,
                          wave_mean_feedback=False, verbose=False)

    # No-WMFI: oscillation period at 25 km (below turning level)
    iz_25 = np.argmin(np.abs(z_km - 25.0))
    psi_amp_nowmfi = np.abs(out_nowmfi["psi_time"][iz_25, :])
    from models.hm76 import estimate_period_days_fft_band
    T_nowmfi, _ = estimate_period_days_fft_band(
        psi_amp_nowmfi[30:200], dt_days=1.0, day_min=10.0, day_max=80.0)
    print(f"  No-WMFI |ψ| vacillation period (25 km, days 30–200): T ≈ {T_nowmfi:.1f} d")

    # Full WMFI: post-burst period (will be modified by changed mean flow)
    iz_32 = np.argmin(np.abs(z_km - 32.0))
    psi_amp_full = np.abs(out["psi_time"][iz_25, :])
    T_full, _ = estimate_period_days_fft_band(
        psi_amp_full[50:200], dt_days=1.0, day_min=10.0, day_max=80.0)
    print(f"  Full WMFI |ψ| vacillation period (25 km, days 50–200): T ≈ {T_full:.1f} d")
    print(f"  (Note: full-WMFI period is modified by mean-flow changes after burst)")

    # Use no-WMFI as the primary comparison
    T_observed = T_nowmfi

    # ================================================================
    # FIGURE 1: Free-mode vertical structures
    # ================================================================
    print("\n--- Plotting free-mode structures ---")

    fig, axes = plt.subplots(1, N_MODES, figsize=(3.5 * N_MODES, 7), sharey=True)
    if N_MODES == 1:
        axes = [axes]

    for rank, ax in enumerate(axes):
        j = order[rank]
        mode = modes[:, j]
        # Rotate so that max amplitude is real
        idx_max = np.argmax(np.abs(mode))
        mode_rot = mode * np.exp(-1j * np.angle(mode[idx_max]))

        ax.plot(np.real(mode_rot), z_km, "C0", lw=2, label="Re")
        ax.plot(np.imag(mode_rot), z_km, "C1", lw=1.5, ls="--", label="Im")
        ax.plot(np.abs(mode_rot), z_km, "k", lw=1, ls=":", alpha=0.5, label="|A|")
        ax.axvline(0, color="k", lw=0.5)

        # Mark turning levels
        for zt in turning_heights:
            ax.axhline(zt, color="green", lw=1, ls="--", alpha=0.7)

        info = mode_info[rank]
        ax.set_title(f"Mode {rank}\n"
                     f"T={info['T_days']:.1f} d\n"
                     f"decay={info['decay_days']:.1f} d\n"
                     f"nodes={info['nodes']}",
                     fontsize=10)
        ax.set_xlim(-1.2, 1.2)
        if rank == 0:
            ax.set_ylabel("Height (km)")
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls=":", alpha=0.3)

    axes[0].set_ylim(10, 80)
    fig.suptitle(rf"Free Rossby modes — background $\bar{{u}}$ at day {T_PROFILE_DAY}"
                 f" (burst case, $h_b={HB:.0f}$ m, $\\tau={TAU_BURST_DAYS:.0f}$ d)\n"
                 f"Green dashes = $m^2=0$ turning levels",
                 fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "step2_free_modes_structure.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # ================================================================
    # FIGURE 2: Mode structures overlaid with m²(z) and ū(z)
    # ================================================================
    print("--- Plotting modes vs m² and ū ---")

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))

    # Panel 1: ū(z) profiles
    ax = axes[0]
    ax.plot(u_initial, z_km, "k--", lw=1.5, label="Initial (day 0)")
    ax.plot(u_profile, z_km, "C0", lw=2.5, label=f"Day {T_PROFILE_DAY}")
    for zt in turning_heights:
        ax.axhline(zt, color="green", lw=1, ls="--", alpha=0.7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\bar{u}$ (m/s)")
    ax.set_ylabel("Height (km)")
    ax.set_title(r"(a)  Background wind $\bar{u}(z)$")
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.3)
    ax.set_ylim(10, 80)

    # Panel 2: m²(z)
    ax = axes[1]
    ax.plot(m2_profile * 1e8, z_km, "C0", lw=2.5)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.fill_betweenx(z_km, 0, m2_profile * 1e8,
                     where=(m2_profile * 1e8 > 0), alpha=0.2, color="C0",
                     label=r"$m^2 > 0$ (propagation)")
    ax.fill_betweenx(z_km, 0, m2_profile * 1e8,
                     where=(m2_profile * 1e8 < 0), alpha=0.2, color="C3",
                     label=r"$m^2 < 0$ (evanescent)")
    for zt in turning_heights:
        ax.axhline(zt, color="green", lw=1.5, ls="--",
                   label=f"Turning level: {zt:.1f} km" if zt == turning_heights[0] else "")
    ax.set_xlabel(r"$m^2$ ($10^{-8}$ m$^{-2}$)")
    ax.set_title(rf"(b)  $m^2(z)$ at day {T_PROFILE_DAY}")
    ax.set_xlim(-2, 3)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    ax.set_ylim(10, 80)

    # Panel 3: Leading 3 mode amplitudes overlaid
    ax = axes[2]
    colors = ["C0", "C1", "C2"]
    for rank in range(min(3, N_MODES)):
        j = order[rank]
        mode = modes[:, j]
        amp = np.abs(mode)
        info = mode_info[rank]
        ax.plot(amp, z_km, colors[rank], lw=2,
                label=f"Mode {rank}: T={info['T_days']:.1f} d")
    for zt in turning_heights:
        ax.axhline(zt, color="green", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Mode amplitude |A(z)|")
    ax.set_title("(c)  Leading free-mode amplitudes")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    ax.set_ylim(10, 80)

    fig.suptitle(rf"Free-mode analysis — day {T_PROFILE_DAY} background wind"
                 f" ($h_b={HB:.0f}$ m, $\\tau={TAU_BURST_DAYS:.0f}$ d)",
                 fontsize=12, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "step2_modes_vs_m2.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # ================================================================
    # FIGURE 3: Period comparison
    # ================================================================
    print("--- Plotting period comparison ---")

    fig, ax = plt.subplots(figsize=(8, 5))

    periods = [info["T_days"] for info in mode_info]
    decays = [info["decay_days"] for info in mode_info]
    ranks = np.arange(N_MODES)

    # Bar chart of mode periods
    bars = ax.bar(ranks, periods, color="C0", alpha=0.7, edgecolor="C0")
    ax.set_xlabel("Mode rank")
    ax.set_ylabel("Period (days)")
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"Mode {r}" for r in ranks], fontsize=9)

    # Observed vacillation period as horizontal lines
    ax.axhline(T_observed, color="C3", lw=2, ls="--",
               label=rf"No-WMFI vacillation: {T_observed:.1f} d")

    # Annotate decay times on bars
    for rank, (bar, decay) in enumerate(zip(bars, decays)):
        if np.isfinite(decay):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"τ_d={decay:.0f}d", ha="center", va="bottom", fontsize=8)

    ax.set_title(rf"Free-mode periods vs observed vacillation — "
                 f"day {T_PROFILE_DAY} background")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.3, axis="y")
    ax.set_ylim(0, max(periods[:N_MODES]) * 1.3)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "step2_period_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Background wind from day {T_PROFILE_DAY} of burst case (hb={HB}, tau={TAU_BURST_DAYS}d)")
    print(f"Turning levels: {[f'{h:.1f} km' for h in turning_heights]}")
    print(f"\nFree-mode periods (first {N_MODES}):")
    for info in mode_info:
        print(f"  Mode {info['rank']}: T = {info['T_days']:.1f} d, "
              f"decay = {info['decay_days']:.1f} d, nodes = {info['nodes']}, "
              f"centroid = {info['centroid_km']:.1f} km")
    print(f"\nObserved no-WMFI vacillation period (25 km, days 30–200): {T_observed:.1f} d")
    print(f"Observed full-WMFI vacillation period (25 km, days 50–200): {T_full:.1f} d")
    best_match = min(mode_info, key=lambda x: abs(x["T_days"] - T_observed))
    print(f"\nBest period match: Mode {best_match['rank']} "
          f"(T = {best_match['T_days']:.1f} d, Δ = {abs(best_match['T_days'] - T_observed):.1f} d)")
    print(f"\nOutput saved to: {OUT_DIR}/")
    print("Done.")
