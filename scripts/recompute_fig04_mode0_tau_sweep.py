"""
scripts/recompute_fig04_mode0_tau_sweep.py
==========================================
Reproducible computation of the Mode-0 characteristic amplitude for Figure 4.

Runs 14 no-WMFI experiments at hb=30 m for 1000 days each, then computes the
N-weighted Mode-0 projection on the ramp-subtracted residual and reports the
time-mean |a0| over a well-defined post-equilibrium window.

Definitions (authoritative for the manuscript):
  - Stationary wave: iterative fixed-point solver on day-0 U0(z).
    LBC: psi_stat(z_b) = g*hb/f0.  UBC: psi_stat(z_t) = 0.
  - Residual: psi_res(z,t) = psi(z,t) - (1 - exp(-t/tau)) * psi_stat(z).
  - Eigenmode: generalized eigenproblem M*phi = omega*N*phi on 69 interior pts,
    alpha_on=False, day-0 background.  Mode 0 normalized: max|phi0| = 1.
  - Projection: a0(t) = (phi0_int^T N psi_res_int(t)) / (phi0_int^T N phi0_int).
  - Characteristic amplitude:
      t_start = max(100 d, 5*tau)
      |a0|_char = mean(|a0(t)|) for t in [t_start, 1000] days.

Output:
    output/data/hm_model/step2_tau_sweep_recomputed_1000d.npz

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/recompute_fig04_mode0_tau_sweep.py
"""

import sys
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hm76 import (
    run_hm76, solve_hm76_eigenmodes, default_hm76_background,
    hm76_constants, compute_betae_for_eigen,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
HB = 30.0
TAU_VALUES = np.array([1., 2., 3., 5., 7., 10., 15., 20., 30., 40., 50., 60., 80., 100.])
N_DAYS = 1000
S = 2.0
DZ = 1000.0
IMAX = 71
DT = 360.0 * 15.0  # 5400 s
ALPHA_ON = False
WAVE_MEAN_FEEDBACK = False

DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Compute the authoritative stationary wave (tau-independent)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Recomputing Mode-0 tau sweep — 1000-day, fully reproducible")
print("=" * 70)
print()

const = hm76_constants(s=S, dz=DZ)
k = const["k"]; l = const["l"]; eps = const["eps"]
f0 = const["f0"]; ensq = const["ensq"]; h0 = const["h0"]
g = 9.81
lrmsq = f0**2 / (4 * ensq * h0**2)
r_coeff = ensq * DZ**2 / f0**2
p_coeff = 2.0 + r_coeff * (k**2 + l**2 + lrmsq)

z = np.arange(IMAX) * DZ + 10000.0
z_km = z / 1000.0
u0 = default_hm76_background(z)
betae = np.real(compute_betae_for_eigen(z, u0, const))

psi_bc = HB * g / f0  # lower boundary condition

# Iterative fixed-point solver
print("Computing iterative stationary solution...")
psi = psi_bc * (1 - np.arange(IMAX) / (IMAX - 1))  # initial guess
for iteration in range(500):
    q = np.zeros(IMAX)
    for i in range(1, IMAX - 1):
        q[i] = -betae[i] * psi[i] / (eps * u0[i])
    ak = np.zeros(IMAX)
    bk = np.zeros(IMAX)
    for i in range(IMAX - 2, 0, -1):
        ak[i - 1] = -1.0 / (ak[i] - p_coeff)
        bk[i - 1] = (r_coeff * q[i] - bk[i]) / (ak[i] - p_coeff)
    psi_new = np.zeros(IMAX)
    psi_new[0] = psi_bc
    for i in range(IMAX - 1):
        psi_new[i + 1] = ak[i] * psi_new[i] + bk[i]
    residual = np.max(np.abs(psi_new - psi))
    if residual < 1e-10:
        print(f"  Converged in {iteration + 1} iterations (residual = {residual:.2e})")
        break
    psi = psi_new.copy()
psi_stat = psi.copy()

psi_stat_max = float(np.max(np.abs(psi_stat)))
print(f"  psi_stat[0] = {psi_stat[0]:.6f} (= g*hb/f0 = {psi_bc:.6f})")
print(f"  psi_stat[-1] = {psi_stat[-1]:.10f}")
print(f"  max|psi_stat| = {psi_stat_max:.6f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Compute eigenmodes (day-0 background, alpha_on=False)
# ══════════════════════════════════════════════════════════════════════════════
print("Solving eigenvalue problem...")
eig_out = solve_hm76_eigenmodes(s=S, dz=DZ, imax=IMAX,
                                 alpha_on=ALPHA_ON, u_bg=u0, n_print=3)

N_mat = np.real(eig_out["Nmat"])  # 69x69 real matrix
j0 = eig_out["order"][0]
phi0_full = np.real(eig_out["modes"][:, j0])  # (71,), boundaries = 0
omega0 = np.real(eig_out["omega"][j0])
T0 = 2.0 * np.pi / np.abs(omega0) / 86400.0
c0 = omega0 / k

# Verify normalization
assert abs(np.max(np.abs(phi0_full)) - 1.0) < 1e-12, "Mode 0 not normalized to max=1"
assert abs(phi0_full[0]) < 1e-12, "Mode 0 LBC not zero"
assert abs(phi0_full[-1]) < 1e-12, "Mode 0 UBC not zero"

# Interior points for projection
phi0_int = phi0_full[1:-1]  # (69,)
N_phi0 = N_mat @ phi0_int
norm_N0 = phi0_int @ N_mat @ phi0_int

print(f"\n  Mode 0: T0 = {T0:.4f} d, c0 = {c0:.4f} m/s")
print(f"  norm_N0 (phi0^T N phi0) = {norm_N0:.6e}")
print(f"  phi0 boundaries: [{phi0_full[0]:.2e}, {phi0_full[-1]:.2e}]")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Run 14 no-WMFI experiments and compute projections
# ══════════════════════════════════════════════════════════════════════════════
n_tau = len(TAU_VALUES)
a0_char = np.zeros(n_tau)
a0_std = np.zeros(n_tau)
a0_cv = np.zeros(n_tau)
t_start_arr = np.zeros(n_tau)
a0_max_arr = np.zeros(n_tau)
frac_explained = np.zeros(n_tau)

print(f"Running {n_tau} no-WMFI experiments (n_days={N_DAYS})...")
print("-" * 70)

t_wall_start = time.time()

for itau, tau_d in enumerate(TAU_VALUES):
    tau_sec = tau_d * 86400.0
    t_start = max(100.0, 5.0 * tau_d)
    t_start_arr[itau] = t_start

    # Run model
    out = run_hm76(HB, tau_sec, s=S, dz=DZ, imax=IMAX, dt=DT,
                   n_days=N_DAYS, alpha_on=ALPHA_ON,
                   wave_mean_feedback=WAVE_MEAN_FEEDBACK,
                   verbose=False)

    psi_time = out["psi_time"].astype(np.complex128)  # (71, 1000)
    nt = psi_time.shape[1]
    t_days = np.arange(nt, dtype=float)

    # Ramp-subtracted residual
    ramp = 1.0 - np.exp(-t_days / tau_d)
    psi_stat_t = psi_stat[:, np.newaxis] * ramp[np.newaxis, :]
    psi_res = psi_time - psi_stat_t

    # N-weighted projection (interior points only)
    a0 = np.array([(N_phi0 @ psi_res[1:-1, tt]) / norm_N0 for tt in range(nt)])

    # Characteristic amplitude over [t_start, 1000]
    mask = t_days >= t_start
    a0_abs = np.abs(a0[mask])
    a0_char[itau] = float(np.mean(a0_abs))
    a0_std[itau] = float(np.std(a0_abs))
    a0_cv[itau] = a0_std[itau] / a0_char[itau] if a0_char[itau] > 0 else 0.0
    a0_max_arr[itau] = float(np.max(np.abs(a0)))

    # Explained variance over same window (full z-t)
    recon = phi0_full[:, np.newaxis] * a0[np.newaxis, :]
    var_res = np.sum(np.abs(psi_res[:, mask])**2)
    unexpl = np.sum(np.abs(psi_res[:, mask] - recon[:, mask])**2)
    frac_explained[itau] = 1.0 - unexpl / var_res if var_res > 0 else 0.0

    elapsed = time.time() - t_wall_start
    print(f"  tau={tau_d:5.0f} d | t_start={t_start:5.0f} | "
          f"|a0|_char={a0_char[itau]/1e6:.4f}×10⁶ | "
          f"CV={a0_cv[itau]*100:.2f}% | "
          f"VE={frac_explained[itau]*100:.2f}% | "
          f"({elapsed:.0f}s)")

print("-" * 70)
elapsed_total = time.time() - t_wall_start
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Summary and verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  hb = {HB} m, n_days = {N_DAYS}, alpha_on = {ALPHA_ON}")
print(f"  Mode 0: T0 = {T0:.4f} d, c0 = {c0:.4f} m/s")
print(f"  Stationary: max|psi_stat| = {psi_stat_max/1e6:.6f} × 10⁶ m²/s (tau-INDEPENDENT)")
print(f"  Projection: N-weighted on 69 interior points")
print(f"  Normalization: max|phi0| = 1")
print()

# Check monotonic decrease
monotonic = np.all(np.diff(a0_char) < 0)
print(f"  Monotonic decrease of |a0|_char with tau: {monotonic}")
if not monotonic:
    for i in range(len(a0_char) - 1):
        if a0_char[i + 1] >= a0_char[i]:
            print(f"    WARNING: a0_char[{i}] ({a0_char[i]:.0f}) <= a0_char[{i+1}] ({a0_char[i+1]:.0f})")

# Check all CVs
print(f"  Max CV across all cases: {np.max(a0_cv)*100:.2f}%")
print(f"  All CVs < 5%: {np.all(a0_cv < 0.05)}")
print()

# Compare with old values
old_path = os.path.join(DATA_DIR, "step2_tau_sweep.npz")
if os.path.exists(old_path):
    old = np.load(old_path)
    old_a0 = old["a0_steady"]
    print("  Old vs New comparison:")
    print(f"  {'tau':>5} {'old':>12} {'new':>12} {'abs_diff':>10} {'rel%':>7}")
    for i, tau in enumerate(TAU_VALUES):
        diff = a0_char[i] - old_a0[i]
        rel = diff / old_a0[i] * 100
        print(f"  {tau:5.0f} {old_a0[i]:12.0f} {a0_char[i]:12.0f} {diff:10.0f} {rel:7.1f}%")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Save
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(DATA_DIR, "step2_tau_sweep_recomputed_1000d.npz")
np.savez_compressed(
    out_path,
    # Core data
    tau_values=TAU_VALUES,
    hb=HB,
    a0_char=a0_char,
    a0_std=a0_std,
    a0_cv=a0_cv,
    a0_max=a0_max_arr,
    t_start=t_start_arr,
    frac_explained=frac_explained,
    # Stationary wave
    psi_stat=psi_stat,
    psi_stat_max=psi_stat_max,
    # Mode parameters
    T_mode0=T0,
    c_mode0=c0,
    omega0=omega0,
    phi0=phi0_full,
    norm_N0=norm_N0,
    # Configuration metadata
    n_days=N_DAYS,
    dt_seconds=DT,
    s=S,
    dz=DZ,
    imax=IMAX,
    alpha_on=ALPHA_ON,
    wave_mean_feedback=WAVE_MEAN_FEEDBACK,
    # Definition metadata
    definition="a0_char = mean(|a0(t)|) for t in [t_start, n_days] where t_start = max(100, 5*tau)",
    projection="N-weighted: (phi0_int^T N psi_res_int) / (phi0_int^T N phi0_int)",
    normalization="max|phi0| = 1",
    residual="psi_res = psi - (1-exp(-t/tau))*psi_stat_iterative",
    stationary_solver="iterative fixed-point on day-0 U0(z), LBC=g*hb/f0, UBC=0",
)
print(f"Saved: {out_path}")
print("\nDone.")
