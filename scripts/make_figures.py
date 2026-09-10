"""
scripts/make_figures.py
=======================
Generate all manuscript figures required by the steering file.

Target filenames (steering file Section 8):
  fig01_hb_tau_phase_diagram.pdf      -- hb-tau burst-transition map
  fig02_interference_paired_cases.pdf -- paired tau=10.91 vs 10.92 cases
  fig03_free_mode_projection.pdf      -- free-mode projection phase
  figA1_hm_free_modes.pdf             -- HM76 free eigenmodes
  figA2_full_nowmfi.pdf               -- full vs nowmfi u(z,t) + |psi|(z,t)
  figA3_m2_diagnostics.pdf            -- m² contour diagnostics

Usage:
  cd /nas/winds-home/smliu01/hm_interference
  python scripts/make_figures.py

All figures are saved to manuscript/figures/.
Intermediate model output is cached to output/data/hm_model/ as .npz files
so that re-running the script does not require re-running the model.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# ── path setup so 'models' and 'plot' are importable ───────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import (
    run_hm76,
    run_hm76_stationary,
    solve_hm76_eigenmodes,
    get_eigenmode_info,
    project_psi_time_onto_eigenmode,
    period_from_complex_phase,
    compute_m2,
)
from plot.plot_hm_model import (
    plot_hm76_eigenmodes,
    plot_hm76_eigenmode_phase,
    plot_u_zt_2x2,
    plot_psi_zt_2x2,
    plot_m2_contour,
    plot_phase_vs_free_reference,
    plot_recoeff_vs_free_marks,
)

# ── output directories ──────────────────────────────────────────────────────
FIG_DIR  = os.path.join(ROOT, "manuscript", "figures")
DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── shared experiment configuration (from cleaned_v4 notebook) ─────────────
S        = 2.0
DZ       = 1000.0
IMAX     = 71
DT       = 360.0 * 15.0
HB       = 27.0           # forcing amplitude (m)
N_DAYS   = 400
ALPHA    = False           # alpha_on=False (no radiative damping on wave)

# tau pair for paired-case and projection figures
TAU_DAYS_LIST = [10.91, 10.92]

# stationary reference (hb=1, long spin-up so it's fully converged)
HB_STAT       = 1.0
TAU_STAT_DAYS = 40.0
N_DAYS_STAT   = 800

# ── helper: stable experiment key ──────────────────────────────────────────
def exp_key(tau_days, feedback_tag):
    return f"tau{tau_days:.2f}_{feedback_tag}".replace(".", "p")

EXP_KEYS = [
    exp_key(tau, fb)
    for tau in TAU_DAYS_LIST
    for fb in ("nowmfi", "full")
]


# ============================================================
# 1. Run or load model experiments
# ============================================================

def _npz_path(name):
    return os.path.join(DATA_DIR, f"{name}.npz")


def _save_out(out, name):
    """Save run_hm76 output dict to npz (complex arrays stored as real+imag)."""
    flat = {}
    for k, v in out.items():
        v = np.asarray(v) if not isinstance(v, (bool, float, int, np.ndarray)) else v
        if isinstance(v, np.ndarray) and np.iscomplexobj(v):
            flat[f"{k}__re"] = np.real(v)
            flat[f"{k}__im"] = np.imag(v)
        elif isinstance(v, np.ndarray):
            flat[k] = v
        else:
            flat[k] = np.array(v)
    np.savez_compressed(_npz_path(name), **flat)


def _load_out(name):
    """Load run_hm76 output dict from npz."""
    data = np.load(_npz_path(name), allow_pickle=False)
    out = {}
    re_keys = {k[:-4] for k in data.files if k.endswith("__re")}
    for k in data.files:
        if k.endswith("__re"):
            base = k[:-4]
            out[base] = data[k] + 1j * data[f"{base}__im"]
        elif k.endswith("__im"):
            pass  # already handled above
        else:
            v = data[k]
            out[k] = float(v) if v.ndim == 0 else v
    return out


def get_or_run(name, run_fn):
    """Return cached model output, or run and cache it."""
    path = _npz_path(name)
    if os.path.exists(path):
        print(f"  loading cached: {name}")
        return _load_out(name)
    print(f"  running model:  {name}")
    out = run_fn()
    _save_out(out, name)
    return out


print("=" * 60)
print("Step 1: running / loading model experiments")
print("=" * 60)

experiments = {}
for tau_days in TAU_DAYS_LIST:
    tau_sec = tau_days * 86400.0
    for fb_tag, fb_kwargs in [
        ("nowmfi", dict(wave_mean_feedback=False, mean_flow_coupling=0.0)),
        ("full",   dict(wave_mean_feedback=True,  mean_flow_coupling=1.0)),
    ]:
        key = exp_key(tau_days, fb_tag)
        out = get_or_run(
            key,
            lambda t=tau_sec, kw=fb_kwargs: run_hm76(
                HB, t, s=S, dz=DZ, imax=IMAX, dt=DT,
                n_days=N_DAYS, alpha_on=ALPHA, **kw, verbose=True,
            ),
        )
        experiments[key] = dict(
            out=out,
            key=key,
            tau_days=tau_days,
            feedback_tag=fb_tag,
        )

# Stationary reference run (needed for Fig 3 projection)
tau_stat_sec = TAU_STAT_DAYS * 86400.0
out_stat = get_or_run(
    "stationary_ref",
    lambda: run_hm76_stationary(
        HB_STAT, tau_stat_sec, s=S, dz=DZ, imax=IMAX, dt=DT,
        n_days=N_DAYS_STAT, alpha_on=True, verbose=True,
    ),
)
# Scale stationary psi up to match hb=HB
psi_stationary = out_stat["psi_stationary"] * (HB / HB_STAT)


# ============================================================
# 2. Eigenvalue analysis (needed for Fig 3 and Fig A1)
# ============================================================

print("\n" + "=" * 60)
print("Step 2: eigenvalue analysis")
print("=" * 60)

eig_nodamp = solve_hm76_eigenmodes(s=S, dz=DZ, imax=IMAX, alpha_on=False, n_print=5)
eig_damp   = solve_hm76_eigenmodes(s=S, dz=DZ, imax=IMAX, alpha_on=True,  n_print=5)

eigen_info = get_eigenmode_info(eig_nodamp, mode_rank=0)
T_free = eigen_info["period_days"]
print(f"\nGravest free mode (no damping): T_free = {T_free:.2f} days, "
      f"c = {eigen_info['c_phase']:.3f} m/s")


# ============================================================
# 3. Free-mode projection (needed for Fig 3)
# ============================================================

print("\n" + "=" * 60)
print("Step 3: free-mode projection")
print("=" * 60)

mode_projections = {}
period_rows = []

for key in EXP_KEYS:
    out = experiments[key]["out"]
    proj = project_psi_time_onto_eigenmode(
        out, eig_nodamp, mode_rank=0,
        psi_stationary=psi_stationary,
        t0_day=20, t1_day=200,
    )
    mode_projections[key] = proj
    ph_result = period_from_complex_phase(proj["t_window"], proj["coeff_window"])
    T_phase = ph_result["period_days"]
    print(f"  {key}: phase period = {T_phase:.2f} d  (free-mode = {T_free:.2f} d)")
    period_rows.append(dict(
        key=key,
        tau_days=experiments[key]["tau_days"],
        feedback=experiments[key]["feedback_tag"],
        period_days=T_phase,
        free_mode_period_days=T_free,
    ))

# Build a simple DataFrame-like object (list of dicts → pandas if available)
try:
    import pandas as pd
    period_table = pd.DataFrame(period_rows)
except ImportError:
    period_table = period_rows   # fallback: plain list


# ============================================================
# Figure A1: HM76 free eigenmodes
# ============================================================

print("\n" + "=" * 60)
print("Generating figA1_hm_free_modes.png")
print("=" * 60)

panel_labels_A1 = ["(a)", "(b)"]
panel_titles_A1 = ["No damping", "With damping"]

# Line styles: rank 0 (gravest, used in projection) drawn thicker
lw_rank  = [2.5, 1.5, 1.5, 1.5]
colors_A1 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True, sharex=True)
fig.subplots_adjust(wspace=0.08, top=0.90, bottom=0.10)

for ax, (eig, plabel, ptitle) in zip(axes, [
    (eig_nodamp, panel_labels_A1[0], panel_titles_A1[0]),
    (eig_damp,   panel_labels_A1[1], panel_titles_A1[1]),
]):
    z_km = eig["z"] / 1000.0
    for rank, j in enumerate(eig["order"][:4]):
        A = eig["modes"][:, j].copy()
        idx = np.argmax(np.abs(A))
        A = A * np.exp(-1j * np.angle(A[idx]))
        wr = np.real(eig["omega"][j])
        T  = 2.0 * np.pi / np.abs(wr) / 86400.0 if np.abs(wr) > 0 else np.inf
        c  = wr / eig["k"]
        suffix = " (gravest, used in Fig.\,3)" if rank == 0 else ""
        ax.plot(np.real(A), z_km,
                color=colors_A1[rank],
                lw=lw_rank[rank],
                label=f"mode {rank+1}: $T$={T:.1f} d, $c$={c:.1f} m/s{suffix}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"Re$(A)$ (normalized)", fontsize=15)
    ax.set_title(f"{plabel} {ptitle}", fontsize=14, loc="left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=11, loc="upper left")

axes[0].set_ylabel("Height (km)", fontsize=15)

fig.suptitle(r"HM76 free-mode eigenmodes, $s=2$", fontsize=13)
fig.savefig(os.path.join(FIG_DIR, "figA1_hm_free_modes.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved figA1_hm_free_modes.png")


# ============================================================
# Figure A2: full vs nowmfi — u(z,t), shared colorbar
# Layout: rows = full / nowmfi;  cols = tau=10.91 (burst) / tau=10.92 (no burst)
# ============================================================

print("\n" + "=" * 60)
print("Generating figA2_full_nowmfi.png")
print("=" * 60)

_A2_DAY0, _A2_DAY1 = 150, 300
_A2_ZMIN, _A2_ZMAX = 10, 80
_A2_LEVS = np.linspace(-80, 60, 29)
_A2_LABELS = [["(a)", "(b)"], ["(c)", "(d)"]]
_A2_FB_ORDER = ["full", "nowmfi"]
_A2_FB_DISP  = {"full": "full WMFI", "nowmfi": "no WMFI"}

fig, axes = plt.subplots(
    2, 2, figsize=(12, 8.5),
    sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.06, "hspace": 0.18},
)

mappable = None
for row, fb in enumerate(_A2_FB_ORDER):
    for col, tau_days in enumerate(TAU_DAYS_LIST):
        key = exp_key(tau_days, fb)
        out = experiments[key]["out"]
        ud   = out["ud"]
        z_km = out["z"] / 1000.0
        nt   = ud.shape[1]
        t_d  = np.arange(nt, dtype=float)

        it = (t_d >= _A2_DAY0) & (t_d <= _A2_DAY1)
        iz = (z_km >= _A2_ZMIN) & (z_km <= _A2_ZMAX)
        tt = t_d[it]; zz = z_km[iz]
        uu = ud[np.ix_(iz, it)]

        ax = axes[row, col]
        cf = ax.contourf(tt, zz, uu, levels=_A2_LEVS, cmap="RdBu_r", extend="both")
        mappable = cf
        ax.grid(True, ls=":", alpha=0.25)

        # Panel label and subtitle
        burst_tag = "burst" if (tau_days == TAU_DAYS_LIST[0] and fb == "full") else "no burst"
        ax.set_title(
            rf"{_A2_LABELS[row][col]}  $\tau = {tau_days:.2f}$ d, {_A2_FB_DISP[fb]}"
            + (f"  ({burst_tag})" if fb == "full" else ""),
            fontsize=13, loc="left",
        )

        # Axis labels only on edges
        if row == 1:
            ax.set_xlabel("Time (days)", fontsize=15)
        if col == 0:
            ax.set_ylabel("Height (km)", fontsize=15)

# Shared colorbar on the right
cbar = fig.colorbar(
    mappable, ax=axes.ravel().tolist(),
    shrink=0.92, pad=0.02, aspect=30,
)
cbar.set_label(r"$\bar{u}$ (m s$^{-1}$)", fontsize=13)

fig.suptitle(
    r"Mean flow $\bar{u}(z,t)$: full vs no-WMFI,  $h_b = 27$ m",
    fontsize=13, y=0.97,
)
fig.savefig(os.path.join(FIG_DIR, "figA2_full_nowmfi.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved figA2_full_nowmfi.png")


# ============================================================
# Figure A3: m² diagnostics — Hovmöller + daily vertical profiles
# Two rows: burst (tau=10.91, full) / no-burst (tau=10.92, full)
# Each row: left = m²(z,t) contour; right = grid of daily m²(z) profiles
# ============================================================

print("\n" + "=" * 60)
print("Generating figA3_m2_diagnostics.png")
print("=" * 60)

import matplotlib.gridspec as gridspec

_A3_ZMIN, _A3_ZMAX = 10, 80
_A3_DAY0_HOV, _A3_DAY1_HOV = 150, 250  # Hovmöller time window
_A3_PROF_DAYS = np.arange(200, 220, 1)  # daily profile days
_A3_NPROF = len(_A3_PROF_DAYS)
_A3_NCOLS_PROF = 6  # profiles per row in the sub-grid
_A3_NROWS_PROF = int(np.ceil(_A3_NPROF / _A3_NCOLS_PROF))

_A3_CASES = [
    (TAU_DAYS_LIST[0], "full", "burst"),     # tau=10.91, full
    (TAU_DAYS_LIST[1], "full", "no burst"),  # tau=10.92, full
]

fig = plt.figure(figsize=(18, 11))
outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.30, top=0.93, bottom=0.05)

for row_idx, (tau_days, fb, burst_label) in enumerate(_A3_CASES):
    key = exp_key(tau_days, fb)
    out = experiments[key]["out"]
    m2  = compute_m2(out, c=0.0, s=S)
    m2_plot = m2 * 1e8  # units: 10^-8 m^-2
    z_km = out["z"] / 1000.0
    nt   = out["ud"].shape[1]
    t_d  = np.arange(nt, dtype=float)

    iz = (z_km >= _A3_ZMIN) & (z_km <= _A3_ZMAX)
    zz = z_km[iz]

    # Inner grid: left = Hovmöller (1 col), right = profile grid
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[row_idx],
        width_ratios=[2.0, 2.2], wspace=0.10,
    )

    # ── Left: m²(z,t) Hovmöller ──
    ax_hov = fig.add_subplot(inner[0, 0])
    it_hov = (t_d >= _A3_DAY0_HOV) & (t_d <= _A3_DAY1_HOV)
    tt_hov = t_d[it_hov]
    mm_hov = m2_plot[np.ix_(iz, it_hov)]
    levs = np.linspace(-4, 4, 33)
    cf = ax_hov.contourf(tt_hov, zz, mm_hov, levels=levs, cmap="RdBu_r", extend="both")
    ax_hov.contour(tt_hov, zz, mm_hov, levels=[0.0], colors="k", linewidths=1.2)
    ax_hov.set_xlabel("Time (days)", fontsize=9)
    ax_hov.set_ylabel("Height (km)", fontsize=9)
    panel_letter = "(a)" if row_idx == 0 else "(c)"
    ax_hov.set_title(
        rf"{panel_letter}  $m^2(z,t)$: $\tau = {tau_days:.2f}$ d, {burst_label}",
        fontsize=9.5, loc="left",
    )
    ax_hov.grid(True, ls=":", alpha=0.3)
    cbar = fig.colorbar(cf, ax=ax_hov, shrink=0.85, pad=0.02)
    cbar.set_label(r"$m^2$ ($10^{-8}$ m$^{-2}$)", fontsize=8)

    # ── Right: daily m²(z) profile sub-grid ──
    # Add an extra top row for the panel title
    inner_prof = gridspec.GridSpecFromSubplotSpec(
        _A3_NROWS_PROF + 1, _A3_NCOLS_PROF,
        subplot_spec=inner[0, 1],
        hspace=0.5, wspace=0.25,
        height_ratios=[0.15] + [1] * _A3_NROWS_PROF,
    )
    panel_letter_prof = "(b)" if row_idx == 0 else "(d)"

    # Title in the extra top row (merged across all columns)
    ax_title = fig.add_subplot(inner_prof[0, :])
    ax_title.set_axis_off()
    ax_title.text(
        0.0, 0.0,
        rf"{panel_letter_prof}  Daily $m^2(z)$ profiles, days {int(_A3_PROF_DAYS[0])}–{int(_A3_PROF_DAYS[-1])}",
        fontsize=9.5, va="bottom", ha="left",
        transform=ax_title.transAxes,
    )

    for pidx, day in enumerate(_A3_PROF_DAYS):
        r = pidx // _A3_NCOLS_PROF + 1  # +1 to skip the title row
        c = pidx % _A3_NCOLS_PROF
        ax_p = fig.add_subplot(inner_prof[r, c])

        iday = int(round(day))
        if iday < nt:
            prof = m2_plot[iz, iday]
            ax_p.plot(prof, zz, color="#1f77b4", lw=1.3)
            ax_p.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
            # Mark m²=0 turning levels with red dots
            for k in range(len(prof) - 1):
                if prof[k] * prof[k + 1] < 0:
                    z_cross = zz[k] + (zz[k+1] - zz[k]) * (-prof[k]) / (prof[k+1] - prof[k])
                    ax_p.plot(0, z_cross, "ro", ms=4, zorder=5)

        ax_p.set_xlim(-5, 5)
        ax_p.set_ylim(_A3_ZMIN, _A3_ZMAX)
        ax_p.set_title(f"day {day:.0f}", fontsize=7, pad=2)
        ax_p.tick_params(labelsize=6)
        if r == _A3_NROWS_PROF:
            ax_p.set_xlabel(r"$m^2$", fontsize=7)
        else:
            ax_p.set_xticklabels([])
        if c == 0:
            ax_p.set_ylabel("km", fontsize=7)
        else:
            ax_p.set_yticklabels([])
        ax_p.grid(True, ls=":", alpha=0.3)

fig.suptitle(
    r"Stationary-wave $m^2$ diagnostics ($c=0$, $s=2$, $h_b=27$ m)",
    fontsize=13, y=0.97,
)
fig.savefig(os.path.join(FIG_DIR, "figA3_m2_diagnostics.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved figA3_m2_diagnostics.png")


# ============================================================
# Figure 2: paired interference cases
# 4 panels: total wave amplitude, phase alignment C_phi,
# interference term, mean-flow response — at z=32 km
# Burst case: tau=10.91 d  |  No-burst case: tau=10.92 d
# ============================================================

print("\n" + "=" * 60)
print("Generating fig02_interference_paired_cases.png")
print("=" * 60)

Z_DIAG_KM = 32.0
iz_diag = np.argmin(np.abs(
    experiments[exp_key(TAU_DAYS_LIST[0], "full")]["out"]["z"] / 1000.0 - Z_DIAG_KM
))

# Stationary reference psi at diagnostic altitude (scaled to hb=HB)
psi_s_scalar = psi_stationary[iz_diag]

def _interf_ts(out_exp):
    """Return (t_d, A_total, C_phi, interf_term, u_diag) at iz_diag."""
    psi_tot = out_exp["psi_time"][iz_diag, :]
    psi_F   = psi_tot - psi_s_scalar
    A_tot   = np.abs(psi_tot)
    A_S     = np.abs(psi_s_scalar)
    A_F     = np.abs(psi_F)
    C_phi   = np.cos(np.angle(psi_s_scalar) - np.angle(psi_F))
    interf  = 2.0 * A_S * A_F * C_phi
    u_d     = out_exp["ud"][iz_diag, :]
    t_d     = np.arange(psi_tot.shape[0], dtype=float)
    return t_d, A_tot, C_phi, interf, u_d

out91_full = experiments[exp_key(TAU_DAYS_LIST[0], "full")]["out"]  # tau=10.91 burst
out92_full = experiments[exp_key(TAU_DAYS_LIST[1], "full")]["out"]  # tau=10.92 no-burst

t91, A91, Cphi91, interf91, u91 = _interf_ts(out91_full)
t92, A92, Cphi92, interf92, u92 = _interf_ts(out92_full)

T0_f2, T1_f2 = 0, 250
mask91 = (t91 >= T0_f2) & (t91 <= T1_f2)

# Key event times (burst case only)
u_srch = u91.copy(); u_srch[:150] = np.inf
t_onset_f2   = float(t91[np.where(u_srch < 20)[0][0]])
mask_pre_f2  = mask91 & (t91 < t_onset_f2)
mask_post_f2 = mask91 & (t91 >= t_onset_f2)
t_trigger_f2 = float(t91[mask_pre_f2][np.argmax(interf91[mask_pre_f2])])
t_peak_f2    = float(t91[mask_post_f2][np.argmax(A91[mask_post_f2])])
print(f"  t_trigger={t_trigger_f2:.0f}d, t_onset={t_onset_f2:.0f}d, t_peak={t_peak_f2:.0f}d")

colors_f2 = {"91": "#1f77b4", "92": "#ff7f0e"}
lw_f2 = 1.8
vkw_trig = dict(color="#2ca02c", lw=1.3, ls="--", alpha=0.85, zorder=2)
vkw_peak = dict(color="#9467bd", lw=1.3, ls="--", alpha=0.85, zorder=2)

def _add_vlines_f2(ax):
    ax.axvline(t_trigger_f2, **vkw_trig)
    ax.axvline(t_peak_f2,    **vkw_peak)

fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
fig.subplots_adjust(hspace=0.22, top=0.91, bottom=0.07)

# Panel (a): total wave amplitude
ax = axes[0]
ax.plot(t91[mask91], A91[mask91], color=colors_f2["91"], lw=lw_f2,
        label=r"$\tau$=10.91 d (burst)")
ax.plot(t92[mask91], A92[mask91], color=colors_f2["92"], lw=lw_f2,
        label=r"$\tau$=10.92 d (no burst)")
ax.axhline(float(np.abs(psi_s_scalar)), color="k", lw=1.2, ls=":", alpha=0.8,
           label=r"stationary $|\Psi_s|$")
_add_vlines_f2(ax)
ax.set_title(rf"(a) Total wave amplitude $|\Psi|$ at $z = {Z_DIAG_KM:.0f}$ km",
             fontsize=14, loc="left")
ax.set_ylabel("Amplitude", fontsize=15)
ax.legend(fontsize=12, loc="upper left", framealpha=0.85)
ax.grid(True, ls=":", alpha=0.35)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ylo, yhi = ax.get_ylim()
ax.text(t_trigger_f2 + 1.5, ylo + (yhi - ylo) * 0.18,
        "max interf./\ndecel. onset", fontsize=10,
        color=vkw_trig["color"], va="bottom", ha="left")
ax.text(t_peak_f2 + 1.5, ylo + (yhi - ylo) * 0.05,
        r"peak $|\Psi|$", fontsize=10,
        color=vkw_peak["color"], va="bottom", ha="left")

# Panel (b): phase alignment
ax = axes[1]
ax.plot(t91[mask91], Cphi91[mask91], color=colors_f2["91"], lw=lw_f2,
        label=r"$\tau$=10.91 d")
ax.plot(t92[mask91], Cphi92[mask91], color=colors_f2["92"], lw=lw_f2,
        label=r"$\tau$=10.92 d")
ax.axhline( 1, color="k", lw=0.8, ls="--", alpha=0.4)
ax.axhline(-1, color="k", lw=0.8, ls="--", alpha=0.4)
ax.axhline( 0, color="k", lw=0.5, alpha=0.2)
_add_vlines_f2(ax)
ax.set_title(r"(b) Phase alignment $C_\phi = \cos(\Delta\phi)$",
             fontsize=14, loc="left")
ax.set_ylabel(r"$C_\phi$", fontsize=15)
ax.set_ylim(-1.25, 1.25)
ax.legend(fontsize=12, loc="lower left", framealpha=0.85)
ax.grid(True, ls=":", alpha=0.35)

# Panel (c): interference term
ax = axes[2]
ax.plot(t91[mask91], interf91[mask91], color=colors_f2["91"], lw=lw_f2,
        label=r"$\tau$=10.91 d")
ax.plot(t92[mask91], interf92[mask91], color=colors_f2["92"], lw=lw_f2,
        label=r"$\tau$=10.92 d")
ax.axhline(0, color="k", lw=0.8)
_add_vlines_f2(ax)
ax.set_title(r"(c) Interference term $2|\Psi_s||\Psi_F|\cos(\Delta\phi)$",
             fontsize=14, loc="left")
ax.set_ylabel(r"$2|\Psi_s||\Psi_F|C_\phi$", fontsize=15)
ax.legend(fontsize=12, loc="upper left", framealpha=0.85)
ax.grid(True, ls=":", alpha=0.35)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Panel (d): mean-flow response
ax = axes[3]
ax.plot(t91[mask91], u91[mask91], color=colors_f2["91"], lw=lw_f2,
        label=r"$\tau$=10.91 d (burst)")
ax.plot(t92[mask91], u92[mask91], color=colors_f2["92"], lw=lw_f2,
        label=r"$\tau$=10.92 d (no burst)")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
_add_vlines_f2(ax)
ax.set_title(rf"(d) Mean-flow response $\bar{{u}}$ at $z = {Z_DIAG_KM:.0f}$ km",
             fontsize=14, loc="left")
ax.set_ylabel(r"$\bar{u}$ (m/s)", fontsize=15)
ax.set_xlabel("Time (days)", fontsize=15)
ax.legend(fontsize=12, loc="lower left", framealpha=0.85)
ax.grid(True, ls=":", alpha=0.35)
ax.set_xlim(T0_f2, T1_f2)

fig.suptitle(
    r"Paired interference cases: $h_b = 27$ m, "
    r"$\tau = 10.91$ d (burst) vs $10.92$ d (no burst)",
    fontsize=11, y=0.975,
)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig02_interference_paired_cases.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved fig02_interference_paired_cases.png")



# ============================================================
# Figure 3: free-mode projection
# ============================================================

print("\n" + "=" * 60)
print("Generating fig03_free_mode_projection.png")
print("=" * 60)

# Panel labels: (a)=top-left, (b)=top-right, (c)=bottom-left, (d)=bottom-right
# Layout: rows = nowmfi / full,  cols = tau=10.91 / tau=10.92
panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.90, bottom=0.08)

for col, tau_days in enumerate(TAU_DAYS_LIST):
    for row, fb in enumerate(("nowmfi", "full")):
        key  = exp_key(tau_days, fb)
        proj = mode_projections[key]
        t_w  = proj["t_window"]
        a    = proj["coeff_window"]
        ph_r = period_from_complex_phase(t_w, a)
        T_phase = ph_r["period_days"]

        ax = axes[row, col]
        label = panel_labels[row][col]
        fb_label = "no WMFI" if fb == "nowmfi" else "full"

        ax.plot(t_w, np.real(a), color="#1f77b4", lw=1.8, label="Re$(a)$")
        ax.plot(t_w, np.imag(a), color="#ff7f0e", lw=1.5, ls="--", label="Im$(a)$")
        ax.axhline(0, color="k", lw=0.8)

        # Mark T_free period with grey dotted lines
        t0 = t_w[0]
        for n in range(1, 8):
            tm = t0 + n * T_free
            if tm <= t_w[-1]:
                ax.axvline(tm, color="grey", ls=":", lw=0.8, alpha=0.7)

        # Panel title with label
        ax.set_title(
            rf"{label} $\tau$={tau_days:.2f} d, {fb_label} — "
            rf"$T_\mathrm{{phase}}$={T_phase:.1f} d, $T_\mathrm{{free}}$={T_free:.1f} d",
            fontsize=12, loc="left"
        )
        ax.set_xlabel("Time (days)", fontsize=15)
        ax.set_ylabel("Projected coefficient", fontsize=14)
        ax.legend(fontsize=12, loc="upper right")
        ax.grid(True, ls=":", alpha=0.4)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.suptitle(
    r"Free-mode projection coefficient $a(t) = \langle A,\,\psi_\mathrm{res}\rangle$"
    r"  (grey dotted lines every $T_\mathrm{free}$)",
    fontsize=12,
)
fig.savefig(os.path.join(FIG_DIR, "fig03_free_mode_projection.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved fig03_free_mode_projection.png")


# ============================================================
# Figure 1: hb-tau burst-transition diagram
# Uses pre-computed high-resolution sweep (hb_tau_sweep_v3long.npz)
# produced by:
#   python scripts/run_sweep_fig01.py          (41×61 grid, 300-day runs)
#   python scripts/run_sweep_boundary_long.py  (boundary cells, 1000-day runs)
# ============================================================

print("\n" + "=" * 60)
print("Generating fig01_hb_tau_phase_diagram.png")
print("=" * 60)

import matplotlib.colors as mcolors

sweep_v3long = _npz_path("hb_tau_sweep_v3long")
if not os.path.exists(sweep_v3long):
    raise FileNotFoundError(
        f"High-resolution sweep not found: {sweep_v3long}\n"
        "Run scripts/run_sweep_fig01.py and scripts/run_sweep_boundary_long.py first."
    )

sw = np.load(sweep_v3long)
burst_map_f1   = sw["burst_map"]    # shape (41 hb, 61 tau)
min_u32_map_f1 = sw["min_u32_map"]
hb_vals_f1     = sw["hb_values"]    # 11..51, step 1
tau_vals_f1    = sw["tau_values"]   # 1..61,  step 1

print(f"  grid: {len(hb_vals_f1)} hb × {len(tau_vals_f1)} tau, "
      f"burst={np.sum(burst_map_f1==1)}/{burst_map_f1.size}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel A: scatter grid + transition boundary ─────────────────────────────
ax = axes[0]
color_no  = "#aec6e8"
color_yes = "#f4a460"
ms = 18

for i, hb in enumerate(hb_vals_f1):
    for j, tau in enumerate(tau_vals_f1):
        c = color_yes if burst_map_f1[i, j] == 1 else color_no
        ax.scatter(hb, tau, c=c, s=ms, zorder=3, edgecolors="none")

# Transition boundary: for each tau row, connect leftmost burst hb - 0.5
boundary_hb, boundary_tau = [], []
for j, tau in enumerate(tau_vals_f1):
    col = burst_map_f1[:, j]
    bi  = np.where(col == 1)[0]
    if len(bi) > 0:
        boundary_hb.append(hb_vals_f1[bi[0]] - 0.5 if bi[0] > 0 else hb_vals_f1[0] - 0.5)
        boundary_tau.append(tau)
if boundary_hb:
    ax.plot(boundary_hb, boundary_tau, "k--", linewidth=1.8, zorder=5)

ax.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_no,
               markersize=8, label="no burst"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_yes,
               markersize=8, label="burst"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.8,
               label="transition boundary"),
    ],
    loc="upper left", framealpha=0.9, fontsize=12,
)
ax.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
ax.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
ax.set_title("(a) Burst / no-burst transition", fontsize=14, loc="left")
ax.set_xticks(np.arange(15, 55, 5))
ax.set_yticks(np.arange(5, 65, 10))
ax.set_xlim(hb_vals_f1[0] - 0.5, hb_vals_f1[-1] + 0.5)
ax.set_ylim(tau_vals_f1[0] - 0.5, tau_vals_f1[-1] + 0.5)
ax.grid(True, linestyle=":", alpha=0.3)

# ── Panel B: minimum u at 32 km, asymmetric colormap ────────────────────────
ax2 = axes[1]
vmin_f1 = float(np.nanmin(min_u32_map_f1))
norm_f1 = mcolors.TwoSlopeNorm(vmin=vmin_f1, vcenter=0.0, vmax=40.0)
pcm2 = ax2.pcolormesh(hb_vals_f1, tau_vals_f1, min_u32_map_f1.T,
                      cmap="RdBu", norm=norm_f1, shading="nearest")
ax2.contour(hb_vals_f1, tau_vals_f1, min_u32_map_f1.T,
            levels=[0.0], colors="k", linewidths=2)
fig.colorbar(pcm2, ax=ax2,
             label="Min. zonal wind at z = 32 km (m/s)", extend="min")
ax2.set_xlabel(r"Forcing amplitude $h_b$ (m)", fontsize=15)
ax2.set_ylabel(r"Forcing spin-up time $\tau$ (days)", fontsize=15)
ax2.set_title(r"(b) Minimum mean flow at $z \approx 32$ km", fontsize=14, loc="left")
ax2.set_xticks(np.arange(15, 55, 5))
ax2.set_yticks(np.arange(5, 65, 10))
ax2.set_xlim(hb_vals_f1[0] - 0.5, hb_vals_f1[-1] + 0.5)
ax2.set_ylim(tau_vals_f1[0] - 0.5, tau_vals_f1[-1] + 0.5)
ax2.grid(True, linestyle=":", alpha=0.3)

fig.suptitle(r"HM76 $(h_b,\,\tau)$ phase diagram, $s=2$", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig01_hb_tau_phase_diagram.png"),
            dpi=600, bbox_inches="tight")
plt.close(fig)
print("  saved fig01_hb_tau_phase_diagram.png")


# ============================================================
# Done
# ============================================================

print("\n" + "=" * 60)
print("All figures saved to:", FIG_DIR)
print("=" * 60)
for fname in sorted(os.listdir(FIG_DIR)):
    print(" ", fname)
