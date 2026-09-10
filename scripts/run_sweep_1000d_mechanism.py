"""
scripts/run_sweep_1000d_mechanism.py
=====================================
Full 41×61 = 2501 point, 1000-day HM sweep with online diagnostics.

Uses the PRODUCTION run_hm76 integration logic (same time-stepping,
same constants) but computes diagnostics at both daily and native (1.5-h)
resolution without saving full z-t fields.

Output diagnostics per (hb, tau):
  Regime / wind:
    min_u32_300, min_u32_1000, burst_300, burst_1000, t_rev_native
  Legacy-compatible amplitude (daily-sampled, matching fig04_Amax_sweep_300d):
    Amax_300_legacy, Amax_1000_legacy
  Legacy-compatible propagation (daily-sampled, matching hb_tau_Mmax_sweep):
    Mmax_300_legacy, Mmax_1000_legacy
  Native-resolution amplitude & propagation (every 1.5-h step):
    Amax_300_native, Amax_1000_native, Mmax_300_native, Mmax_1000_native
  Native-resolution timing:
    t_A_native, t_open_native (days, float; NaN if never crossed)
  State at threshold crossings:
    A_at_open_native, M_at_Across_native
  Pre-reversal propagation (audit):
    Mmax_pre_rev_1000

Production config:
    alpha_on=False, wave_mean_feedback=True, mean_flow_coupling=1.0
    s=2, dz=1000, imax=71, dt=5400 s, n_days=1000

Grid: hb=11..51 (step 1, 41), tau=1..61 (step 1, 61), total 2501

Output: output/data/hm_model/hb_tau_mechanism_1000d.npz

Usage:
    cd /nas/winds-home/smliu01/hm_interference
    python scripts/run_sweep_1000d_mechanism.py
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Configuration ───────────────────────────────────────────────────────────
S = 2.0
DZ = 1000.0
IMAX = 71
DT = 360.0 * 15.0       # 5400 s = 1.5 h
N_DAYS = 1000
ALPHA_ON = False
WAVE_MEAN_FEEDBACK = True
MEAN_FLOW_COUPLING = 1.0

# Diagnostic parameters
BURST_Z_IDX = 22         # z = 32 km
BURST_THRESH = 0.0       # wind reversal threshold
SPINUP_DAYS = 50         # skip first 50 days for burst detection
A_C = 5.76e6             # amplitude threshold (m²/s)
Z1_KM = 25.0             # barrier layer lower bound
Z2_KM = 40.0             # barrier layer upper bound
C_BARRIER = 0.0          # stationary wave (c=0)

# Grid
HB_VALUES = np.arange(11, 52, 1, dtype=float)   # 41 values
TAU_VALUES = np.arange(1, 62, 1, dtype=float)   # 61 values

N_WORKERS = 32
DATA_DIR = os.path.join(ROOT, "output", "data", "hm_model")
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINT_INTERVAL = 200  # save progress every N cases


def _run_one_case(args):
    """
    Run one (hb, tau) case for 1000 days using the PRODUCTION HM76 integration
    (identical to run_hm76 in models/hm76.py) but with online diagnostics
    computed at both daily and native resolution.

    Returns a dict of scalar diagnostics.
    """
    hb, tau_d = args
    tau_sec = tau_d * 86400.0

    # ── HM76 constants (same as production run_hm76) ────────────────────────
    a = 6378000.0
    k = S / (a * np.cos(np.pi / 3.0))
    l = 3.0 / a
    omega_e = 7.29e-5
    g = 9.81
    f0 = 2.0 * omega_e * np.sin(np.pi / 3.0)
    beta = 2.0 * omega_e * np.cos(np.pi / 3.0) / a
    ensq = 4.0e-4
    h0 = 7000.0
    eps = 8.0 / (3.0 * np.pi)
    lrmsq = (f0 * f0) / (4.0 * ensq * h0 * h0)
    r = (ensq * DZ * DZ) / (f0 * f0)
    p = 2.0 + r * (k * k + l * l + lrmsq)
    pu = 2.0 + r * l * l
    vp = 1.0 + 0.5 * DZ / h0
    vm = 1.0 - 0.5 * DZ / h0
    K2 = k ** 2 + l ** 2 + lrmsq  # for m² computation

    steps_per_day = int(3600 * 24.0 / DT)  # 16
    mmax = N_DAYS * steps_per_day
    dt_days = DT / 86400.0  # 0.0625 days per step

    # Barrier layer indices
    z_arr = np.arange(IMAX) * DZ + 10000.0
    z_km = z_arr / 1000.0
    barrier_mask = (z_km >= Z1_KM) & (z_km <= Z2_KM)
    barrier_indices = np.where(barrier_mask)[0]

    # ── Allocate model state ────────────────────────────────────────────────
    u0 = np.zeros((IMAX, 4))
    qy = np.zeros((IMAX, 4))
    psi = np.zeros((IMAX, 4), dtype=np.complex128)
    q = np.zeros((IMAX, 4), dtype=np.complex128)
    ak = np.zeros(IMAX)
    bk = np.zeros(IMAX, dtype=np.complex128)
    ck = np.zeros(IMAX)
    dk = np.zeros(IMAX)
    alpha = np.zeros(IMAX)

    # Initial conditions (same as production)
    for i in range(IMAX):
        if ALPHA_ON:
            alpha[i] = (1.5 + np.tanh((z_arr[i] - 35000.0) / h0)) * 1.0e-6
        u0[i, :] = 12.0 + (z_arr[i] - 10000.0) * (52.0 - 12.0) / 40000.0
    for i in range(1, IMAX - 1):
        qy[i, :] = -l * l * u0[i, 0] + (f0 * f0 / ensq) * (
            (u0[i + 1, 0] + u0[i - 1, 0] - 2.0 * u0[i, 0]) / (DZ * DZ)
            - (u0[i + 1, 0] - u0[i - 1, 0]) / (2.0 * DZ * h0))
    u_bg = u0[:, 0].copy()
    qy_bg = qy[:, 0].copy()

    # ── Online diagnostic accumulators ──────────────────────────────────────
    # Legacy (daily) accumulators
    Amax_300_leg = 0.0
    Amax_1000_leg = 0.0
    Mmax_300_leg = -np.inf
    Mmax_1000_leg = -np.inf

    # Native (every step) accumulators
    Amax_300_nat = 0.0
    Amax_1000_nat = 0.0
    Mmax_300_nat = -np.inf
    Mmax_1000_nat = -np.inf
    Mmax_pre_rev = -np.inf

    # Wind tracking
    min_u32_300 = np.inf
    min_u32_1000 = np.inf

    # Native timing (first crossing, in fractional days)
    t_A_native = np.nan
    t_open_native = np.nan
    t_rev_native = np.nan
    A_at_open = np.nan
    M_at_Across = np.nan

    # Flags
    burst_300 = 0
    burst_1000 = 0
    rev_occurred = False

    day_count = 0  # counts daily outputs

    # ── Main integration loop (PRODUCTION HM76 logic) ───────────────────────
    m = 0
    while m < mmax:
        t = DT * m
        t_days_current = t / 86400.0

        if not WAVE_MEAN_FEEDBACK:
            u0[:, 0] = u_bg; u0[:, 1] = u_bg; u0[:, 2] = u_bg
            qy[:, 0] = qy_bg; qy[:, 1] = qy_bg; qy[:, 2] = qy_bg

        # Wave + mean-flow tendency (AB3)
        for i in range(1, IMAX - 1):
            if WAVE_MEAN_FEEDBACK:
                betae1 = beta - eps * qy[i, 0]
                betae2 = beta - eps * qy[i, 1]
                betae3 = beta - eps * qy[i, 2]
                uadv1 = -(0 + 1j) * k * eps * u0[i, 0] * q[i, 0]
                uadv2 = -(0 + 1j) * k * eps * u0[i, 1] * q[i, 1]
                uadv3 = -(0 + 1j) * k * eps * u0[i, 2] * q[i, 2]
            else:
                betae1 = betae2 = betae3 = beta - eps * qy_bg[i]
                uadv1 = -(0 + 1j) * k * eps * u_bg[i] * q[i, 0]
                uadv2 = -(0 + 1j) * k * eps * u_bg[i] * q[i, 1]
                uadv3 = -(0 + 1j) * k * eps * u_bg[i] * q[i, 2]

            vadv1 = -betae1 * (0 + 1j) * k * psi[i, 0]
            vadv2 = -betae2 * (0 + 1j) * k * psi[i, 1]
            vadv3 = -betae3 * (0 + 1j) * k * psi[i, 2]

            damp0 = (f0 * f0 / ensq) * (0.5 / h0) * alpha[i] * (
                (psi[i + 1, 2] - psi[i - 1, 2]) / (2.0 * DZ) + psi[i, 2] / (2.0 * h0))
            damp1 = -(f0 * f0 / (ensq * DZ * DZ)) * 0.5 * (alpha[i + 1] + alpha[i]) * (
                psi[i + 1, 2] - psi[i, 2])
            damp2 = (f0 * f0 / (ensq * DZ * DZ)) * 0.5 * (alpha[i - 1] + alpha[i]) * (
                psi[i, 2] - psi[i - 1, 2])
            damp3 = -(f0 * f0 / (ensq * DZ * h0)) * 0.125 * (
                (alpha[i + 1] + alpha[i]) * psi[i + 1, 2]
                - (alpha[i - 1] + alpha[i]) * psi[i - 1, 2]
                + (alpha[i + 1] - alpha[i - 1]) * psi[i, 2])

            q[i, 3] = q[i, 2] + (DT / 12.0) * (
                5.0 * uadv1 - 16.0 * uadv2 + 23.0 * uadv3
                + 5.0 * vadv1 - 16.0 * vadv2 + 23.0 * vadv3)
            q[i, 3] += DT * (damp0 + damp1 + damp2 + damp3)

            # Mean-flow PV tendency
            flux1 = 0.5 * l * l * k * eps * (f0 * f0 / ensq) * np.exp(z_arr[i] / h0) * np.imag(
                psi[i, 0] * np.conj(psi[i + 1, 0] + psi[i - 1, 0] - 2.0 * psi[i, 0]) / (DZ * DZ))
            flux2 = 0.5 * l * l * k * eps * (f0 * f0 / ensq) * np.exp(z_arr[i] / h0) * np.imag(
                psi[i, 1] * np.conj(psi[i + 1, 1] + psi[i - 1, 1] - 2.0 * psi[i, 1]) / (DZ * DZ))
            flux3 = 0.5 * l * l * k * eps * (f0 * f0 / ensq) * np.exp(z_arr[i] / h0) * np.imag(
                psi[i, 2] * np.conj(psi[i + 1, 2] + psi[i - 1, 2] - 2.0 * psi[i, 2]) / (DZ * DZ))

            zm = 0.5 * (z_arr[i] + z_arr[i - 1]) / h0
            zp = 0.5 * (z_arr[i] + z_arr[i + 1]) / h0
            udamp1 = -(f0 * f0 / ensq) * np.exp(z_arr[i] / h0) * (1.0 / DZ) * (
                0.5 * (alpha[i + 1] + alpha[i]) * np.exp(-zp) * (
                    (u0[i + 1, 2] - u0[i, 2]) / DZ - 3.0e-3))
            udamp2 = (f0 * f0 / ensq) * np.exp(z_arr[i] / h0) * (1.0 / DZ) * (
                0.5 * (alpha[i - 1] + alpha[i]) * np.exp(-zm) * (
                    (u0[i, 2] - u0[i - 1, 2]) / DZ - 3.0e-3))

            if WAVE_MEAN_FEEDBACK:
                qy[i, 3] = qy[i, 2] + (DT / 12.0) * MEAN_FLOW_COUPLING * (
                    5.0 * flux1 - 16.0 * flux2 + 23.0 * flux3) + DT * (udamp1 + udamp2)
            else:
                qy[i, 3] = qy_bg[i]

        # Invert psi from q
        ak[IMAX - 2] = 0.0
        bk[IMAX - 2] = 0.0 + 0.0j
        for i in range(IMAX - 2, -1, -1):
            ak[i - 1] = -1.0 / (ak[i] - p)
            bk[i - 1] = (r * q[i, 3] - bk[i]) / (ak[i] - p)
        psi[0, 3] = hb * (g / f0) * (1.0 - np.exp(-t / tau_sec))
        for i in range(IMAX - 1):
            psi[i + 1, 3] = ak[i] * psi[i, 3] + bk[i]

        # Invert u0 from qy
        ck[IMAX - 2] = 1.0
        dk[IMAX - 2] = 0.0
        for i in range(IMAX - 2, -1, -1):
            ck[i - 1] = -vp / (ck[i] * vm - pu)
            dk[i - 1] = (r * qy[i, 3] - dk[i] * vm) / (ck[i] * vm - pu)
        u0[0, 3] = 12.0
        for i in range(IMAX - 1):
            u0[i + 1, 3] = ck[i] * u0[i, 3] + dk[i]

        if not WAVE_MEAN_FEEDBACK:
            u0[:, 3] = u_bg
            qy[:, 3] = qy_bg

        # Shift time levels
        qy[:, 0] = qy[:, 1]; qy[:, 1] = qy[:, 2]; qy[:, 2] = qy[:, 3]
        q[:, 0] = q[:, 1]; q[:, 1] = q[:, 2]; q[:, 2] = q[:, 3]
        psi[:, 0] = psi[:, 1]; psi[:, 1] = psi[:, 2]; psi[:, 2] = psi[:, 3]
        u0[:, 0] = u0[:, 1]; u0[:, 1] = u0[:, 2]; u0[:, 2] = u0[:, 3]

        # ── Native-resolution diagnostics (every step) ──────────────────────
        # After shift, psi[:,2] and u0[:,2] are the current state
        A_now = abs(psi[BURST_Z_IDX, 2])
        u32_now = u0[BURST_Z_IDX, 2].real

        # Compute M(t) at native step
        betae_now = beta - eps * qy[:, 2]
        denom_barrier = eps * u0[:, 2].real
        # m² at barrier levels
        M_now = np.inf
        for bi in barrier_indices:
            d_val = denom_barrier[bi]
            if abs(d_val) < 1e-6:
                continue  # skip singular points
            m2_val = (ensq / (f0 * f0)) * (betae_now[bi] / d_val - K2)
            if m2_val < M_now:
                M_now = m2_val
        if np.isinf(M_now):
            M_now = np.nan

        # Time in days for this step (AFTER the advance)
        t_now_days = (m + 1) * dt_days

        # Native Amax/Mmax
        if t_now_days <= 300.0:
            if A_now > Amax_300_nat:
                Amax_300_nat = A_now
            if not np.isnan(M_now) and M_now > Mmax_300_nat:
                Mmax_300_nat = M_now
        if A_now > Amax_1000_nat:
            Amax_1000_nat = A_now
        if not np.isnan(M_now) and M_now > Mmax_1000_nat:
            Mmax_1000_nat = M_now

        # Pre-reversal Mmax
        if not rev_occurred and not np.isnan(M_now) and M_now > Mmax_pre_rev:
            Mmax_pre_rev = M_now

        # Wind tracking
        if t_now_days > SPINUP_DAYS:
            if t_now_days <= 300.0 and u32_now < min_u32_300:
                min_u32_300 = u32_now
            if u32_now < min_u32_1000:
                min_u32_1000 = u32_now

        # Native timing: t_A (first A > Ac)
        if np.isnan(t_A_native) and A_now > A_C:
            t_A_native = t_now_days
            # M at the moment A crosses Ac
            M_at_Across = M_now if not np.isnan(M_now) else np.nan

        # Native timing: t_open (first M > 0)
        if np.isnan(t_open_native) and not np.isnan(M_now) and M_now > 0:
            t_open_native = t_now_days
            # A at the moment barrier opens
            A_at_open = A_now

        # Native timing: t_rev (first u32 < 0 after spinup)
        if not rev_occurred and t_now_days > SPINUP_DAYS and u32_now < BURST_THRESH:
            t_rev_native = t_now_days
            rev_occurred = True

        # ── Daily output diagnostics (legacy-compatible) ────────────────────
        if m % steps_per_day == 0:
            # Matches production run_hm76 convention: output at m=0, 16, 32, ...
            # After the shift, psi[:,2] is the current state.
            day_count += 1

            # Legacy Amax (daily-sampled |psi(32km)|)
            if day_count <= 300:
                if A_now > Amax_300_leg:
                    Amax_300_leg = A_now
            if A_now > Amax_1000_leg:
                Amax_1000_leg = A_now

            # Legacy Mmax (daily-sampled M(t))
            if not np.isnan(M_now):
                if day_count <= 300 and M_now > Mmax_300_leg:
                    Mmax_300_leg = M_now
                if M_now > Mmax_1000_leg:
                    Mmax_1000_leg = M_now

        m += 1

    # ── Post-process ────────────────────────────────────────────────────────
    # Burst flags
    burst_300 = int(min_u32_300 < BURST_THRESH) if np.isfinite(min_u32_300) else 0
    burst_1000 = int(min_u32_1000 < BURST_THRESH) if np.isfinite(min_u32_1000) else 0

    # Handle -inf Mmax (barrier never approached zero)
    if np.isinf(Mmax_300_leg) and Mmax_300_leg < 0:
        Mmax_300_leg = np.nan
    if np.isinf(Mmax_1000_leg) and Mmax_1000_leg < 0:
        Mmax_1000_leg = np.nan
    if np.isinf(Mmax_300_nat) and Mmax_300_nat < 0:
        Mmax_300_nat = np.nan
    if np.isinf(Mmax_1000_nat) and Mmax_1000_nat < 0:
        Mmax_1000_nat = np.nan
    if np.isinf(Mmax_pre_rev) and Mmax_pre_rev < 0:
        Mmax_pre_rev = np.nan

    return dict(
        hb=hb, tau=tau_d,
        min_u32_300=min_u32_300 if np.isfinite(min_u32_300) else np.nan,
        min_u32_1000=min_u32_1000 if np.isfinite(min_u32_1000) else np.nan,
        burst_300=burst_300,
        burst_1000=burst_1000,
        t_rev_native=t_rev_native,
        Amax_300_legacy=Amax_300_leg,
        Amax_1000_legacy=Amax_1000_leg,
        Mmax_300_legacy=Mmax_300_leg,
        Mmax_1000_legacy=Mmax_1000_leg,
        Amax_300_native=Amax_300_nat,
        Amax_1000_native=Amax_1000_nat,
        Mmax_300_native=Mmax_300_nat,
        Mmax_1000_native=Mmax_1000_nat,
        Mmax_pre_rev_1000=Mmax_pre_rev,
        t_A_native=t_A_native,
        t_open_native=t_open_native,
        A_at_open_native=A_at_open,
        M_at_Across_native=M_at_Across,
    )


def main():
    t0 = time.time()
    n_hb = len(HB_VALUES)
    n_tau = len(TAU_VALUES)
    total = n_hb * n_tau
    print(f"1000-day mechanism sweep: {n_hb} x {n_tau} = {total} cases")
    print(f"  n_days={N_DAYS}, dt={DT}s, steps/day={int(86400/DT)}")
    print(f"  workers={N_WORKERS}")
    print(f"  A_c={A_C:.4e}, barrier=[{Z1_KM},{Z2_KM}] km, c={C_BARRIER}")
    print(f"  Estimated time: ~{total * 33 / N_WORKERS / 60:.0f} min")
    print()

    tasks = [(hb, tau) for hb in HB_VALUES for tau in TAU_VALUES]

    # Allocate output arrays
    shape = (n_hb, n_tau)
    results = {
        'min_u32_300': np.full(shape, np.nan),
        'min_u32_1000': np.full(shape, np.nan),
        'burst_300': np.full(shape, -1, dtype=int),
        'burst_1000': np.full(shape, -1, dtype=int),
        't_rev_native': np.full(shape, np.nan),
        'Amax_300_legacy': np.full(shape, np.nan),
        'Amax_1000_legacy': np.full(shape, np.nan),
        'Mmax_300_legacy': np.full(shape, np.nan),
        'Mmax_1000_legacy': np.full(shape, np.nan),
        'Amax_300_native': np.full(shape, np.nan),
        'Amax_1000_native': np.full(shape, np.nan),
        'Mmax_300_native': np.full(shape, np.nan),
        'Mmax_1000_native': np.full(shape, np.nan),
        'Mmax_pre_rev_1000': np.full(shape, np.nan),
        't_A_native': np.full(shape, np.nan),
        't_open_native': np.full(shape, np.nan),
        'A_at_open_native': np.full(shape, np.nan),
        'M_at_Across_native': np.full(shape, np.nan),
    }

    completed = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for res in pool.imap_unordered(_run_one_case, tasks):
            hb, tau_d = res['hb'], res['tau']
            i = int(hb - HB_VALUES[0])
            j = int(tau_d - TAU_VALUES[0])

            for key in results:
                if key in res:
                    results[key][i, j] = res[key]

            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{total} done ({elapsed:.0f}s, ETA {eta:.0f}s)")

            # Checkpoint
            if completed % CHECKPOINT_INTERVAL == 0:
                _save(results, checkpoint=True)

    elapsed = time.time() - t0
    print(f"\nAll {total} cases done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Final save
    _save(results, checkpoint=False)

    # Quick summary
    n_burst_300 = np.sum(results['burst_300'] == 1)
    n_burst_1000 = np.sum(results['burst_1000'] == 1)
    print(f"\n--- Quick summary ---")
    print(f"  burst_300: {n_burst_300}/{total}")
    print(f"  burst_1000: {n_burst_1000}/{total}")
    print(f"  Mmax_1000 > 0: {np.sum(results['Mmax_1000_legacy'] > 0)}/{total}")
    print(f"  Amax_1000 >= Ac: {np.sum(results['Amax_1000_legacy'] >= A_C)}/{total}")


def _save(results, checkpoint=False):
    """Save results to npz."""
    tag = "_checkpoint" if checkpoint else ""
    outpath = os.path.join(DATA_DIR, f"hb_tau_mechanism_1000d{tag}.npz")
    np.savez_compressed(
        outpath,
        hb_values=HB_VALUES,
        tau_values=TAU_VALUES,
        n_days=N_DAYS,
        dt_seconds=DT,
        sampling_hours_native=DT / 3600.0,
        A_c=A_C,
        z_A_km=32.0,
        z1_barrier_km=Z1_KM,
        z2_barrier_km=Z2_KM,
        c_barrier=C_BARRIER,
        alpha_on=ALPHA_ON,
        wave_mean_feedback=WAVE_MEAN_FEEDBACK,
        mean_flow_coupling=MEAN_FLOW_COUPLING,
        burst_z_idx=BURST_Z_IDX,
        burst_thresh=BURST_THRESH,
        spinup_days=SPINUP_DAYS,
        **results,
    )
    if not checkpoint:
        print(f"  Saved: {outpath}")


if __name__ == "__main__":
    main()
