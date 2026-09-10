"""
models/two_level.py
===================
Two-level quasilinear stratospheric wave-mean-flow model.
Based on Noboru Nakamura's June 26, 2026 two-level model notes.

Contents
--------
Data class
    TwoLevelParams

Helper / matrix builders
    damping_rates
    interface_damping_rates
    pv_matrix
    q_hat
    beta_effective
    build_wave_matrices_updated
    mean_flow_matrices
    eddy_mean_forcing
    radiative_equilibrium_forcing

Solvers
    solve_2level_free_modes_updated
    solve_2level_ramped_forcing
    sweep_hb_tau_ramped
    solve_2level_forced_adjustment_updated
    integrate_2level_quasilinear

Diagnostics
    compute_u10_proxy
    compute_mean_flow_tendencies
    diagnose_ssw_like_event
    sweep_hb_tau_for_events
    initial_condition_from_stationary_plus_free
    sweep_initial_free_phase
    run_full_ql_case_for_comparison

Utilities
    summarize_2level_modes
    print_mode_summary
    get_u10_weights
    compute_u10_from_U_array
    classify_ssw_like
    format_hb_for_label
    make_case
"""

import numpy as np
from dataclasses import dataclass
from scipy.linalg import eig
from scipy.integrate import solve_ivp


# ============================================================
# Data class
# ============================================================

@dataclass
class TwoLevelParams:
    """Parameters for the 2+2 level quasilinear stratospheric model."""
    # Geometry / constants
    s: int = 1
    a_earth: float = 6.371e6
    Omega: float = 7.292e-5
    phi0_deg: float = 60.0
    Ly: float = 1.0e7
    # Vertical model parameters
    D: float = 13_000.0
    H: float = 7_000.0
    N0: float = 0.02
    f0: float = 1.0e-4
    # Damping. Scalar tau_days or length-4 list/tuple for levels 0..3.
    tau_days: object = 20.0
    damping: bool = True
    # Projection factor epsilon = 8 / (3*pi)
    eps: float = 8.0 / (3.0 * np.pi)

    @property
    def phi0(self):
        return np.deg2rad(self.phi0_deg)
    @property
    def k(self):
        return self.s / (self.a_earth * np.cos(self.phi0))
    @property
    def l(self):
        return np.pi / self.Ly
    @property
    def beta(self):
        return 2.0 * self.Omega * np.cos(self.phi0) / self.a_earth
    @property
    def K2(self):
        return self.k**2 + self.l**2
    @property
    def alpha(self):
        return np.exp(-self.D / self.H)
    @property
    def a(self):
        return self.alpha**0.5
    @property
    def b(self):
        return self.alpha**(-0.5)
    @property
    def LD2(self):
        return (self.N0**2 * self.D**2) / (self.f0**2)
    @property
    def invLD2(self):
        return 1.0 / self.LD2


def damping_rates(params: TwoLevelParams):
    """Return gamma0, gamma1, gamma2, gamma3 in s⁻¹."""
    if not params.damping:
        return np.zeros(4, dtype=float)
    tau = np.asarray(params.tau_days, dtype=float)
    if tau.ndim == 0:
        tau = np.repeat(float(tau), 4)
    if tau.size != 4:
        raise ValueError("tau_days must be a scalar or a length-4 list/tuple.")
    return 1.0 / (tau * 86400.0)


def interface_damping_rates(params: TwoLevelParams):
    """Return gamma0.5, gamma1.5, gamma2.5 in s⁻¹."""
    g0, g1, g2, g3 = damping_rates(params)
    return 0.5*(g0+g1), 0.5*(g1+g2), 0.5*(g2+g3)


# ============================================================
# Matrix builders
# ============================================================

def pv_matrix(params: TwoLevelParams):
    """PV matrix Mq: q_active = Mq @ [A1, A2] + boundary terms."""
    K2=params.K2; a=params.a; b=params.b; inv=params.invLD2
    return np.array([[-K2-(a+b)*inv, b*inv],
                     [ a*inv,        -K2-(a+b)*inv]], dtype=complex)


def q_hat(A, params: TwoLevelParams, A0=0.0+0.0j, A3=0.0+0.0j):
    """Return qhat = [q1hat, q2hat] including boundary wave amplitudes."""
    A=np.asarray(A,dtype=complex); q=pv_matrix(params)@A
    q[0]+=params.a*params.invLD2*A0; q[1]+=params.b*params.invLD2*A3
    return q


def beta_effective(params: TwoLevelParams, U0, U1, U2, U3):
    """Effective beta at levels 1 and 2, Eq. (14)-(15)."""
    a=params.a; b=params.b; eps=params.eps; l=params.l; inv=params.invLD2
    be1=params.beta+eps*l**2*U1-eps*inv*(a*U0-(a+b)*U1+b*U2)
    be2=params.beta+eps*l**2*U2-eps*inv*(a*U1-(a+b)*U2+b*U3)
    return be1, be2


def build_wave_matrices_updated(params: TwoLevelParams,
                                U1=0.0, U2=0.0, U0=0.0, U3=0.0):
    """
    Build updated fixed-U wave equation Mq dA/dt = L A + f_A0*A0 + f_A3*A3.

    Returns
    -------
    Mq, L, f_A0, f_A3, info
    """
    Mq=pv_matrix(params)
    g05,g15,g25=interface_damping_rates(params)
    a=params.a; b=params.b; inv=params.invLD2; eps=params.eps; k=params.k
    be1,be2=beta_effective(params,U0,U1,U2,U3)
    Rwave=np.array([[(g05*a+g15*b)*inv, -g15*b*inv],
                    [-g15*a*inv, (g15*a+g25*b)*inv]], dtype=complex)
    Umat=np.diag([eps*U1, eps*U2]).astype(complex)
    Bmat=np.diag([be1, be2]).astype(complex)
    L=Rwave-1j*k*(Umat@Mq+Bmat)
    f_A0=np.array([-g05*a*inv-1j*k*eps*U1*a*inv, 0.0], dtype=complex)
    f_A3=np.array([0.0, -g25*b*inv-1j*k*eps*U2*b*inv], dtype=complex)
    info=dict(beta_e1=be1, beta_e2=be2, gamma05=g05, gamma15=g15, gamma25=g25, epsilon=eps)
    return Mq, L, f_A0, f_A3, info


def mean_flow_matrices(params: TwoLevelParams):
    """Mean-flow M and R matrices from Eq. (16)."""
    alpha=params.alpha; l=params.l; LD2=params.LD2
    g05,g15,g25=interface_damping_rates(params)
    c=1.0/(np.sqrt(alpha)*LD2)
    Mmean=np.array([[l**2+c, -c],
                    [-c, l**2+(1.0+alpha)*c]], dtype=float)
    Rmean=c*np.array([[-g15, g15],
                      [alpha*g15, -(alpha*g15+g25)]], dtype=float)
    return Mmean, Rmean


def eddy_mean_forcing(A, params: TwoLevelParams, A0=0.0+0.0j, A3=0.0+0.0j):
    """Eddy forcing F_n = -eps*k*l²/2 * Im(psi_n * q_n^*), Eq. (17)."""
    A=np.asarray(A,dtype=complex); q=q_hat(A,params,A0=A0,A3=A3)
    return np.real(-params.eps*params.k*params.l**2/2.0*np.imag(A*np.conj(q)))


def radiative_equilibrium_forcing(params: TwoLevelParams, UR1=20.0, UR2=10.0):
    """Radiative equilibrium forcing vector in Eq. (16)."""
    alpha=params.alpha; g05,g15,g25=interface_damping_rates(params)
    c=1.0/(np.sqrt(alpha)*params.LD2)
    return c*np.array([g15*(UR1-UR2),
                       -alpha*g15*(UR1-UR2)+g25*UR2], dtype=float)


# ============================================================
# Solvers
# ============================================================

def solve_2level_free_modes_updated(params: TwoLevelParams,
                                    U1=0.0, U2=0.0, U0=0.0, U3=0.0):
    """
    Solve free modes: L e = lambda Mq e.

    Returns
    -------
    lam : eigenvalues sorted by period (shortest first)
    vec : eigenvectors
    info : dict with beta_e, damping rates, epsilon
    """
    Mq,L,_,_,info=build_wave_matrices_updated(params,U1=U1,U2=U2,U0=U0,U3=U3)
    lam,vec=eig(L,Mq)
    omega=np.abs(np.imag(lam))
    period=np.where(omega>0, 2*np.pi/omega/86400.0, np.inf)
    idx=np.argsort(period)
    return lam[idx], vec[:,idx], info


def solve_2level_ramped_forcing(params: TwoLevelParams,
                                hb=1.0+0.0j, A0=0.0+0.0j,
                                U1=0.0, U2=0.0, U0=0.0, U3=0.0,
                                tau_spinup_days=10.0,
                                tmax_days=120, nt=1500):
    """
    Fixed-U forced adjustment with exponentially ramped lower-boundary forcing:
        A3(t) = hb * (1 - exp(-t / tau_spinup))

    This is the two-level analog of the HM76 spin-up experiment.

    The full solution is decomposed as:
        A(t) = A_stat * (1 - exp(-t/tau_s)) + sum_j c_j * exp(lambda_j * t) * e_j

    where A_stat is the steady-state forced response and c_j are the free-mode
    projection coefficients whose magnitudes depend on tau_spinup.

    Parameters
    ----------
    params : TwoLevelParams
    hb : complex   Final lower-boundary forcing amplitude.
    A0 : complex   Upper boundary amplitude (default 0).
    U1, U2, U0, U3 : float   Fixed background winds.
    tau_spinup_days : float   Forcing spin-up e-folding time (days).
    tmax_days : float   Integration time.
    nt : int   Number of output time steps.

    Returns
    -------
    dict with keys:
        t_days, A_t (total), A_stat, A_forced_t (time-dependent forced part),
        A_free_t (free/transient part), free_coeff (projection coefficients),
        lam (eigenvalues), vec (eigenvectors), peak_total_amp,
        peak_total_amp_upper, peak_total_amp_lower,
        stationary_amp_upper, stationary_amp_lower,
        free_amp_upper, free_amp_lower, tau_spinup_days, info.
    """
    tau_s = tau_spinup_days * 86400.0  # convert to seconds
    t_days = np.linspace(0, tmax_days, nt)
    t_sec = t_days * 86400.0

    # Build matrices
    Mq, L, f_A0, f_A3, info = build_wave_matrices_updated(
        params, U1=U1, U2=U2, U0=U0, U3=U3)

    # Stationary solution: L * A_stat = -f (where f = f_A0*A0 + f_A3*hb)
    forcing = f_A0 * A0 + f_A3 * hb
    A_stat = -np.linalg.solve(L, forcing)

    # Free eigenmodes: Mq^{-1} L e_j = lambda_j e_j
    lam, vec, _ = solve_2level_free_modes_updated(
        params, U1=U1, U2=U2, U0=U0, U3=U3)

    # For ramped forcing A3(t) = hb*(1-exp(-t/tau_s)), the particular solution is:
    #   A_particular(t) = A_stat * (1 - exp(-t/tau_s))
    #     only if -1/tau_s is not an eigenvalue. We also need the complementary
    #     (homogeneous) solution to satisfy A(0) = 0.
    #
    # Full solution: A(t) = A_stat - A_stat*exp(-t/tau_s)
    #                       + sum_j c_j * exp(lambda_j * t) * e_j
    #
    # But the particular solution for the exponential forcing term requires solving:
    #   Mq * dA/dt = L*A + f * hb * (1 - exp(-t/tau_s))
    #
    # Let A(t) = A_stat + B(t)*exp(-t/tau_s) + sum_j c_j * exp(lambda_j*t) * e_j
    # where B is a particular solution for the exp(-t/tau_s) forcing.
    #
    # Substituting: Mq*(-B/tau_s)*exp(-t/tau_s) + Mq*sum(c_j*lam_j*exp(lam_j*t)*e_j)
    #   = L*A_stat + L*B*exp(-t/tau_s) + L*sum(c_j*exp(lam_j*t)*e_j)
    #     + f*hb - f*hb*exp(-t/tau_s)
    #
    # Using L*A_stat = -f*hb:
    #   Mq*(-B/tau_s)*exp(-t/tau_s) = L*B*exp(-t/tau_s) - f*hb*exp(-t/tau_s)
    #   => (L + Mq/tau_s) * B = -f*hb = L*A_stat
    #   => B = (L + Mq/tau_s)^{-1} * L * A_stat
    #
    # IC: A(0) = 0 => A_stat + B + sum_j c_j * e_j = 0
    #   => sum_j c_j * e_j = -(A_stat + B)
    #   => c = vec^{-1} * (-(A_stat + B))

    # Particular solution for the exp(-t/tau_s) part
    L_shifted = L + Mq / tau_s
    B = np.linalg.solve(L_shifted, L @ A_stat)

    # Free-mode projection coefficients
    free_coeff = np.linalg.solve(vec, -(A_stat + B))

    # Time evolution
    A_t = np.zeros((nt, 2), dtype=complex)
    A_forced_t = np.zeros((nt, 2), dtype=complex)
    A_free_t = np.zeros((nt, 2), dtype=complex)

    for n, tt in enumerate(t_sec):
        # Forced (particular) part: stationary + exponential ramp
        A_forced = A_stat + B * np.exp(-tt / tau_s)
        # Free (homogeneous) part
        A_free = vec @ (free_coeff * np.exp(lam * tt))
        A_forced_t[n, :] = A_forced
        A_free_t[n, :] = A_free
        A_t[n, :] = A_forced + A_free

    # Peak amplitudes
    total_amp = np.sqrt(np.abs(A_t[:, 0])**2 + np.abs(A_t[:, 1])**2)
    peak_total_amp = float(np.max(total_amp))
    peak_total_amp_upper = float(np.max(np.abs(A_t[:, 0])))
    peak_total_amp_lower = float(np.max(np.abs(A_t[:, 1])))

    return dict(
        t_days=t_days, A_t=A_t, A_stat=A_stat,
        A_forced_t=A_forced_t, A_free_t=A_free_t,
        B=B, free_coeff=free_coeff,
        lam=lam, vec=vec,
        peak_total_amp=peak_total_amp,
        peak_total_amp_upper=peak_total_amp_upper,
        peak_total_amp_lower=peak_total_amp_lower,
        stationary_amp_upper=float(np.abs(A_stat[0])),
        stationary_amp_lower=float(np.abs(A_stat[1])),
        free_amp_upper=float(np.max(np.abs(A_free_t[:, 0]))),
        free_amp_lower=float(np.max(np.abs(A_free_t[:, 1]))),
        tau_spinup_days=tau_spinup_days, info=info,
    )


def sweep_hb_tau_ramped(params: TwoLevelParams,
                         hb_values=None, tau_spinup_values=None,
                         A0=0.0+0.0j,
                         U1=20.0, U2=10.0, U0=None, U3=0.0,
                         tmax_days=120, nt=1500):
    """
    Systematic parameter sweep over (hb, tau_spinup) for the ramped-forcing
    two-level model (fixed background wind, no wave–mean flow interaction).

    This is the two-level analog of the HM76 (hb, tau) sweep.

    Parameters
    ----------
    params : TwoLevelParams   Base parameters (damping, geometry).
    hb_values : array-like   Final forcing amplitudes (m²/s units in streamfunction).
    tau_spinup_values : array-like   Forcing spin-up times in days.
    U1, U2, U0, U3 : float   Fixed background winds (m/s).
    tmax_days : float   Integration time for each run.
    nt : int   Number of time steps per run.

    Returns
    -------
    dict with keys:
        hb_values, tau_spinup_values,
        A_stat_amp_upper, A_stat_amp_lower (nh,) — stationary amplitudes (hb-dependent only),
        free_amp_upper, free_amp_lower (nh, ntau) — peak free-mode amplitude,
        peak_total_amp (nh, ntau) — peak total wave amplitude,
        peak_total_amp_upper, peak_total_amp_lower (nh, ntau),
        free_coeff_0, free_coeff_1 (nh, ntau) — free-mode projection coefficients,
        lam (2,) — eigenvalues (same for all hb since U is fixed),
        vec (2,2) — eigenvectors,
        params, U1, U2, U0, U3, tmax_days, nt.
    """
    if U0 is None:
        U0 = U1
    if hb_values is None:
        hb_values = np.linspace(1e6, 5e8, 30)
    if tau_spinup_values is None:
        tau_spinup_values = np.linspace(1.0, 60.0, 30)

    hb_values = np.asarray(hb_values, dtype=float)
    tau_spinup_values = np.asarray(tau_spinup_values, dtype=float)
    nh = len(hb_values)
    ntau = len(tau_spinup_values)

    # Eigenvalues/vectors are the same for all (hb, tau) since U is fixed
    lam, vec, info = solve_2level_free_modes_updated(
        params, U1=U1, U2=U2, U0=U0, U3=U3)

    # Allocate output arrays
    A_stat_amp_upper = np.zeros(nh)
    A_stat_amp_lower = np.zeros(nh)
    free_amp_upper = np.zeros((nh, ntau))
    free_amp_lower = np.zeros((nh, ntau))
    peak_total_amp = np.zeros((nh, ntau))
    peak_total_amp_upper = np.zeros((nh, ntau))
    peak_total_amp_lower = np.zeros((nh, ntau))
    free_coeff_abs = np.zeros((nh, ntau, 2))

    for i, hb in enumerate(hb_values):
        for j, tau_s in enumerate(tau_spinup_values):
            result = solve_2level_ramped_forcing(
                params, hb=hb + 0j, A0=A0,
                U1=U1, U2=U2, U0=U0, U3=U3,
                tau_spinup_days=tau_s,
                tmax_days=tmax_days, nt=nt)

            if j == 0:
                # Stationary amplitude depends only on hb
                A_stat_amp_upper[i] = result['stationary_amp_upper']
                A_stat_amp_lower[i] = result['stationary_amp_lower']

            free_amp_upper[i, j] = result['free_amp_upper']
            free_amp_lower[i, j] = result['free_amp_lower']
            peak_total_amp[i, j] = result['peak_total_amp']
            peak_total_amp_upper[i, j] = result['peak_total_amp_upper']
            peak_total_amp_lower[i, j] = result['peak_total_amp_lower']
            free_coeff_abs[i, j, :] = np.abs(result['free_coeff'])

    return dict(
        hb_values=hb_values, tau_spinup_values=tau_spinup_values,
        A_stat_amp_upper=A_stat_amp_upper, A_stat_amp_lower=A_stat_amp_lower,
        free_amp_upper=free_amp_upper, free_amp_lower=free_amp_lower,
        peak_total_amp=peak_total_amp,
        peak_total_amp_upper=peak_total_amp_upper,
        peak_total_amp_lower=peak_total_amp_lower,
        free_coeff_abs=free_coeff_abs,
        lam=lam, vec=vec, info=info,
        params=params, U1=U1, U2=U2, U0=U0, U3=U3,
        tmax_days=tmax_days, nt=nt,
    )


def solve_2level_forced_adjustment_updated(params: TwoLevelParams,
                                           hb=1.0+0.0j, A0=0.0+0.0j,
                                           U1=0.0, U2=0.0, U0=0.0, U3=0.0,
                                           tmax_days=120, nt=1500):
    """
    Fixed-U forced adjustment with prescribed lower-boundary forcing A3=hb.

    Returns
    -------
    dict  t_days, A_t, A_stat, A_free_t, lambda, vec, info.
    """
    t_days=np.linspace(0,tmax_days,nt); t_sec=t_days*86400.0
    Mq,L,f_A0,f_A3,info=build_wave_matrices_updated(params,U1=U1,U2=U2,U0=U0,U3=U3)
    forcing=f_A0*A0+f_A3*hb
    A_stat=-np.linalg.solve(L,forcing)
    lam,vec,_=solve_2level_free_modes_updated(params,U1=U1,U2=U2,U0=U0,U3=U3)
    coeff=np.linalg.solve(vec,-A_stat)
    A_t=np.zeros((nt,2),dtype=complex); A_free_t=np.zeros((nt,2),dtype=complex)
    for n,tt in enumerate(t_sec):
        A_free=vec@(coeff*np.exp(lam*tt))
        A_free_t[n,:]=A_free; A_t[n,:]=A_stat+A_free
    return dict(t_days=t_days,A_t=A_t,A_stat=A_stat,A_free_t=A_free_t,
                lam=lam,vec=vec,info=info)


def _rhs_quasilinear(t_sec, y, params, hb, A0, UR1, UR2, U3, wave_mean_flow):
    A=np.array([y[0]+1j*y[1], y[2]+1j*y[3]], dtype=complex)
    U1=float(y[4]); U2=float(y[5]); U0=U1
    Mq,L,f_A0,f_A3,_=build_wave_matrices_updated(params,U1=U1,U2=U2,U0=U0,U3=U3)
    dA_dt=np.linalg.solve(Mq,L@A+f_A0*A0+f_A3*hb)
    if wave_mean_flow:
        Mmean,Rmean=mean_flow_matrices(params)
        Feddy=eddy_mean_forcing(A,params,A0=A0,A3=hb)
        Frad=radiative_equilibrium_forcing(params,UR1=UR1,UR2=UR2)
        dU_dt=np.linalg.solve(Mmean,Rmean@np.array([U1,U2])+Feddy+Frad)
    else:
        dU_dt=np.zeros(2)
    return np.array([np.real(dA_dt[0]),np.imag(dA_dt[0]),
                     np.real(dA_dt[1]),np.imag(dA_dt[1]),
                     dU_dt[0],dU_dt[1]], dtype=float)


def integrate_2level_quasilinear(params: TwoLevelParams,
                                 hb=1.0+0.0j, A0=0.0+0.0j,
                                 A1_init=0.0+0.0j, A2_init=0.0+0.0j,
                                 U1_init=20.0, U2_init=10.0,
                                 UR1=20.0, UR2=10.0, U3=0.0,
                                 tmax_days=300.0, dt_days=0.02,
                                 wave_mean_flow=True,
                                 rtol=1e-7, atol=1e-9):
    """
    Integrate the full quasilinear 2-level model.

    Parameters
    ----------
    wave_mean_flow : bool
        True → full model; False → nowmfi (wave sees fixed U).

    Returns
    -------
    dict  t_days, A_t (nt×2 complex), U_t (nt×2), Feddy_t (nt×2),
          params, hb, A0, UR1, UR2, U3, wave_mean_flow.
    """
    y0=np.array([np.real(A1_init),np.imag(A1_init),
                 np.real(A2_init),np.imag(A2_init),
                 U1_init,U2_init], dtype=float)
    t_eval_days=np.arange(0.0,tmax_days+0.5*dt_days,dt_days)
    t_eval_sec=t_eval_days*86400.0
    sol=solve_ivp(_rhs_quasilinear,
                  t_span=(0.0,tmax_days*86400.0), y0=y0,
                  t_eval=t_eval_sec,
                  args=(params,hb,A0,UR1,UR2,U3,wave_mean_flow),
                  method="RK45", rtol=rtol, atol=atol,
                  max_step=dt_days*86400.0)
    if not sol.success: raise RuntimeError(sol.message)
    A_t=(sol.y[0,:]+1j*sol.y[1,:]); B_t=(sol.y[2,:]+1j*sol.y[3,:])
    U_t=np.vstack([sol.y[4,:],sol.y[5,:]]).T
    nt=len(t_eval_days)
    Feddy_t=np.zeros((nt,2))
    for i in range(nt):
        Feddy_t[i,:]=eddy_mean_forcing([A_t[i],B_t[i]],params,A0=A0,A3=hb)
    return dict(t_days=t_eval_days,A_t=np.vstack([A_t,B_t]).T,U_t=U_t,Feddy_t=Feddy_t,
                params=params,hb=hb,A0=A0,UR1=UR1,UR2=UR2,U3=U3,wave_mean_flow=wave_mean_flow)


# ============================================================
# Diagnostics
# ============================================================

def compute_u10_proxy(result, p_upper=5.0, p_lower=30.0, p_target=10.0):
    """Log-pressure interpolation from U1 (~5 hPa) and U2 (~30 hPa) to 10 hPa."""
    U=result["U_t"]
    w_upper=np.log(p_lower/p_target)/np.log(p_lower/p_upper)
    return w_upper*U[:,0]+(1.0-w_upper)*U[:,1]


def get_u10_weights(p_upper=5.0, p_lower=30.0, p_target=10.0):
    """Return (w_upper, w_lower) for the U10 proxy."""
    w=np.log(p_lower/p_target)/np.log(p_lower/p_upper)
    return w, 1.0-w


def compute_u10_from_U_array(U, p_upper=5.0, p_lower=30.0, p_target=10.0):
    """U: array (nt, 2) with columns U1, U2."""
    w1,w2=get_u10_weights(p_upper=p_upper,p_lower=p_lower,p_target=p_target)
    return w1*U[:,0]+w2*U[:,1]


def compute_mean_flow_tendencies(result):
    """
    Return eddy, radiative, and total tendencies in m/s/day.

    Returns
    -------
    eddy_tendency, rad_tendency, total_tendency : ndarray, each shape (nt, 2)
    """
    params=result["params"]; U=result["U_t"]; Feddy=result["Feddy_t"]
    Mmean,Rmean=mean_flow_matrices(params)
    Frad=radiative_equilibrium_forcing(params,UR1=result["UR1"],UR2=result["UR2"])
    eddy=np.zeros_like(U); rad=np.zeros_like(U); total=np.zeros_like(U)
    for i in range(len(U)):
        eddy[i,:]=np.linalg.solve(Mmean,Feddy[i,:])
        rad[i,:]=np.linalg.solve(Mmean,Rmean@U[i,:]+Frad)
        total[i,:]=eddy[i,:]+rad[i,:]
    spd=86400.0
    return eddy*spd, rad*spd, total*spd


def diagnose_ssw_like_event(result, wind_metric="U10",
                             reversal_threshold=0.0, drop_threshold=10.0,
                             wind_level=None, threshold=None):
    """
    Diagnose SSW-like behavior.

    Parameters
    ----------
    wind_metric : {"U1", "U2", "U10"}
    reversal_threshold : float   Default 0 m/s.
    drop_threshold     : float   Default 10 m/s.

    Returns
    -------
    dict  event, reverses, large_drop, onset_type, onset_day, u_initial, u_min,
          t_min, max_drop, max_Aamp, t_max_Aamp, max_abs_F, t_max_abs_F,
          wind_metric, reversal_threshold, drop_threshold.
    """
    if wind_level is not None: wind_metric=wind_level
    if threshold is not None: reversal_threshold=threshold
    t=result["t_days"]; U=result["U_t"]; A=result["A_t"]; F=result["Feddy_t"]
    if wind_metric=="U1": u=U[:,0]
    elif wind_metric=="U2": u=U[:,1]
    elif wind_metric=="U10": u=compute_u10_proxy(result)
    else: raise ValueError("wind_metric must be 'U1', 'U2', or 'U10'.")
    u0=float(u[0]); u_min=float(np.min(u)); i_min=int(np.argmin(u))
    max_drop=u0-u_min; reverses=bool(np.any(u<reversal_threshold)); large_drop=bool(max_drop>=drop_threshold)
    if reverses:
        idx=int(np.where(u<reversal_threshold)[0][0]); onset_day=float(t[idx]); onset_type="reversal"
    elif large_drop:
        idx=int(np.where(u<=u0-drop_threshold)[0][0]); onset_day=float(t[idx]); onset_type="large_drop"
    else:
        onset_day=np.nan; onset_type="none"
    Aamp=np.sqrt(np.abs(A[:,0])**2+np.abs(A[:,1])**2)
    f_diag=F[:,0] if wind_metric=="U1" else (F[:,1] if wind_metric=="U2" else np.sqrt(F[:,0]**2+F[:,1]**2))
    return dict(event=bool(reverses or large_drop),reverses=bool(reverses),large_drop=bool(large_drop),
                onset_type=onset_type,onset_day=onset_day,u_initial=u0,u_min=u_min,
                t_min=float(t[i_min]),max_drop=float(max_drop),
                max_Aamp=float(np.max(Aamp)),t_max_Aamp=float(t[np.argmax(Aamp)]),
                max_abs_F=float(np.max(np.abs(f_diag))),t_max_abs_F=float(t[np.argmax(np.abs(f_diag))]),
                wind_metric=wind_metric,reversal_threshold=float(reversal_threshold),drop_threshold=float(drop_threshold))


def sweep_hb_tau_for_events(s=1, hb_values=None, tau_values=None,
                             U1_init=20.0, U2_init=10.0, UR1=20.0, UR2=10.0, U3=0.0,
                             tmax_days=300, dt_days=0.2, wind_metric="U10",
                             reversal_threshold=0.0, drop_threshold=10.0,
                             wind_level=None, threshold=None):
    """
    Sweep lower-boundary forcing h_b and damping timescale tau.

    Returns
    -------
    dict  hb_values, tau_values, event_map, reverse_map, drop_map,
          minU_map, drop_amount_map, onset_map, maxA_map, maxF_map,
          diagnostics, s, wind_metric, reversal_threshold, drop_threshold.
    """
    if wind_level is not None: wind_metric=wind_level
    if threshold is not None: reversal_threshold=threshold
    if hb_values is None: hb_values=np.logspace(6,9,13)
    if tau_values is None: tau_values=np.array([5,7.5,10,15,20,30,40,60,80])
    nh,nt=len(hb_values),len(tau_values)
    event_map=np.zeros((nh,nt),dtype=int); reverse_map=np.zeros((nh,nt),dtype=int)
    minU_map=np.zeros((nh,nt)); drop_amount_map=np.zeros((nh,nt))
    onset_map=np.full((nh,nt),np.nan); maxA_map=np.zeros((nh,nt)); maxF_map=np.zeros((nh,nt))
    diagnostics={}
    for i,hb in enumerate(hb_values):
        for j,tau in enumerate(tau_values):
            params=TwoLevelParams(s=s,tau_days=float(tau),damping=True)
            result=integrate_2level_quasilinear(params,hb=hb+0j,U1_init=U1_init,U2_init=U2_init,
                                                UR1=UR1,UR2=UR2,U3=U3,tmax_days=tmax_days,
                                                dt_days=dt_days,wave_mean_flow=True,rtol=1e-6,atol=1e-8)
            diag=diagnose_ssw_like_event(result,wind_metric=wind_metric,
                                         reversal_threshold=reversal_threshold,drop_threshold=drop_threshold)
            event_map[i,j]=int(diag["event"]); reverse_map[i,j]=int(diag["reverses"])
            minU_map[i,j]=diag["u_min"]; drop_amount_map[i,j]=diag["max_drop"]
            onset_map[i,j]=diag["onset_day"]; maxA_map[i,j]=diag["max_Aamp"]; maxF_map[i,j]=diag["max_abs_F"]
            diagnostics[(float(hb),float(tau))]=diag
            print(f"s={s}, hb={hb:.2e}, tau={tau:.1f}: event={diag['event']}, "
                  f"type={diag['onset_type']}, min {wind_metric}={diag['u_min']:.2f}, onset={diag['onset_day']}")
    return dict(hb_values=np.array(hb_values,dtype=float),tau_values=np.array(tau_values,dtype=float),
                event_map=event_map,reverse_map=reverse_map,minU_map=minU_map,
                drop_amount_map=drop_amount_map,onset_map=onset_map,maxA_map=maxA_map,maxF_map=maxF_map,
                diagnostics=diagnostics,s=s,wind_metric=wind_metric,
                reversal_threshold=reversal_threshold,drop_threshold=drop_threshold)


def initial_condition_from_stationary_plus_free(params, hb, U1, U2,
                                                U0=None, U3=0.0,
                                                free_mode_index=0,
                                                free_amp_fraction=0.5, phase=0.0):
    """
    Build initial [A1, A2] as stationary response + one free eigenmode.

    Returns
    -------
    A1_init, A2_init, A_stat, eigenvector e, eigenvalue lam.
    """
    if U0 is None: U0=U1
    fixed=solve_2level_forced_adjustment_updated(params,hb=hb+0j,U1=U1,U2=U2,U0=U0,U3=U3,tmax_days=1,nt=2)
    A_stat=fixed["A_stat"]
    lam,vec,_=solve_2level_free_modes_updated(params,U1=U1,U2=U2,U0=U0,U3=U3)
    e=vec[:,free_mode_index].copy(); e=e/np.max(np.abs(e))
    free_amp=free_amp_fraction*np.max(np.abs(A_stat))
    A_init=A_stat+free_amp*e*np.exp(1j*phase)
    return A_init[0], A_init[1], A_stat, e, lam[free_mode_index]


def sweep_initial_free_phase(s=1, hb=1e8, tau=20.0, free_mode_index=0,
                              free_amp_fraction=0.5, nphase=24,
                              U1_init=20.0, U2_init=10.0, UR1=20.0, UR2=10.0, U3=0.0,
                              tmax_days=300, dt_days=0.1, wind_metric="U10",
                              reversal_threshold=0.0, drop_threshold=10.0):
    """
    Sweep the initial phase of one free mode added to the stationary response.

    Returns
    -------
    dict  phases, event, reverses, large_drop, minU, onset, max_drop,
          results, diagnostics, s, hb, tau, free_mode_index, free_amp_fraction,
          wind_metric, reversal_threshold, drop_threshold.
    """
    params=TwoLevelParams(s=s,tau_days=tau,damping=True)
    phases=np.linspace(0,2*np.pi,nphase,endpoint=False)
    event=np.zeros(nphase,dtype=int); reverses=np.zeros(nphase,dtype=int)
    large_drop=np.zeros(nphase,dtype=int); minU=np.zeros(nphase)
    onset=np.full(nphase,np.nan); max_drop=np.zeros(nphase)
    results=[]; diagnostics=[]
    for i,phase in enumerate(phases):
        A1,A2,_,_,_=initial_condition_from_stationary_plus_free(
            params,hb=hb,U1=U1_init,U2=U2_init,U0=U1_init,U3=U3,
            free_mode_index=free_mode_index,free_amp_fraction=free_amp_fraction,phase=phase)
        result=integrate_2level_quasilinear(params,hb=hb+0j,A1_init=A1,A2_init=A2,
                                            U1_init=U1_init,U2_init=U2_init,UR1=UR1,UR2=UR2,U3=U3,
                                            tmax_days=tmax_days,dt_days=dt_days,wave_mean_flow=True)
        diag=diagnose_ssw_like_event(result,wind_metric=wind_metric,
                                     reversal_threshold=reversal_threshold,drop_threshold=drop_threshold)
        event[i]=int(diag["event"]); reverses[i]=int(diag["reverses"]); large_drop[i]=int(diag["large_drop"])
        minU[i]=diag["u_min"]; onset[i]=diag["onset_day"]; max_drop[i]=diag["max_drop"]
        results.append(result); diagnostics.append(diag)
        print(f"phase={phase:.2f}: event={diag['event']}, type={diag['onset_type']}, "
              f"min {wind_metric}={diag['u_min']:.2f}, onset={diag['onset_day']}")
    return dict(phases=phases,event=event,reverses=reverses,large_drop=large_drop,
                minU=minU,onset=onset,max_drop=max_drop,results=results,diagnostics=diagnostics,
                s=s,hb=hb,tau=tau,free_mode_index=free_mode_index,free_amp_fraction=free_amp_fraction,
                wind_metric=wind_metric,reversal_threshold=reversal_threshold,drop_threshold=drop_threshold)


def run_full_ql_case_for_comparison(s, hb, tau, U1_init=20.0, U2_init=10.0,
                                    UR1=20.0, UR2=10.0, U3=0.0,
                                    tmax_days=150.0, dt_days=0.1):
    """
    Run one full QL case and package diagnostics for multi-case comparison plots.

    Returns
    -------
    dict  s, hb, tau, result, t, A1_abs/real, A2_abs/real, U1, U2, U10,
          eddy_tend_U1/U2/U10, diag, status.
    """
    params=TwoLevelParams(s=s,tau_days=float(tau),damping=True)
    result=integrate_2level_quasilinear(params,hb=hb+0j,U1_init=U1_init,U2_init=U2_init,
                                        UR1=UR1,UR2=UR2,U3=U3,tmax_days=tmax_days,
                                        dt_days=dt_days,wave_mean_flow=True)
    t=result["t_days"]; A=result["A_t"]; U=result["U_t"]
    U10=compute_u10_proxy(result)
    eddy_tend,_,_=compute_mean_flow_tendencies(result)
    w1,w2=get_u10_weights()
    eddy_tend_U10=w1*eddy_tend[:,0]+w2*eddy_tend[:,1]
    diag=diagnose_ssw_like_event(result,wind_metric="U10",drop_threshold=10.0)
    return dict(s=s,hb=hb,tau=tau,result=result,t=t,
                A1_abs=np.abs(A[:,0]),A2_abs=np.abs(A[:,1]),
                A1_real=np.real(A[:,0]),A2_real=np.real(A[:,1]),
                U1=U[:,0],U2=U[:,1],U10=U10,
                eddy_tend_U1=eddy_tend[:,0],eddy_tend_U2=eddy_tend[:,1],
                eddy_tend_U10=eddy_tend_U10,diag=diag,status=classify_ssw_like(diag))


# ============================================================
# Utilities
# ============================================================

def summarize_2level_modes(lam, vec):
    """Return a list of dicts describing each eigenmode."""
    rows=[]
    for j in range(len(lam)):
        v=vec[:,j].copy(); v=v/np.max(np.abs(v))
        omega=np.abs(np.imag(lam[j]))
        T=np.inf if omega==0 else 2*np.pi/omega/86400.0
        decay=-1.0/np.real(lam[j])/86400.0 if np.real(lam[j])<0 else np.inf
        ph=np.angle(v[1]/v[0])
        rows.append(dict(mode=j,lam=lam[j],period_days=T,decay_days=decay,
                         psi1=v[0],psi2=v[1],phase_diff_psi2_minus_psi1=ph,
                         type="barotropic-like" if np.cos(ph)>0 else "baroclinic-like"))
    return rows


def print_mode_summary(rows):
    for r in rows:
        print(f"mode: {r['mode']}  type: {r['type']}  lam: {r['lam']:.4e}")
        print(f"  period: {r['period_days']:.2f} d  decay: {r['decay_days']:.2f} d")
        print(f"  psi1: {r['psi1']:.4e}  psi2: {r['psi2']:.4e}  phase diff: {r['phase_diff_psi2_minus_psi1']:.3f}")
        print()


def classify_ssw_like(diag):
    if diag["reverses"]: return "U10 reversal"
    elif diag["event"]: return "strong weakening"
    return "no SSW-like"


def format_hb_for_label(hb):
    """Format hb as LaTeX string, e.g. 1.2e8 → r'1.2\\times 10^8'."""
    hb=float(hb)
    if hb==0: return "0"
    exp=int(np.floor(np.log10(abs(hb)))); mant=hb/(10**exp)
    return rf"10^{exp}" if np.isclose(mant,1.0) else rf"{mant:g}\times 10^{exp}"


def make_case(hb, tau, color=None, linestyle="-", linewidth=2.0):
    """Build a case dict with auto-formatted label for plot_compare_cases."""
    return dict(hb=float(hb),tau=float(tau),
                label=rf"$h_b={format_hb_for_label(hb)},\ \tau={tau:g}$ d",
                color=color,linestyle=linestyle,linewidth=linewidth)
