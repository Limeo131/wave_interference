"""
models/hm76.py
==============
Holton-Mass 1976 (HM76) model functions.

Contents
--------
Time integrators
    run_hm76              -- full / nowmfi unified integrator
    run_hm76_stationary   -- stationary forced run (background frozen)
    run_hm76_transient    -- transient free-wave run

Diagnostics
    compute_m2            -- refractive index squared m²(z,t)
    compute_Ucrit_fields  -- critical-level wind fields

Transient mode extraction
    extract_transient_mode

Period estimation utilities
    estimate_period_days_fft_band
    estimate_c_from_series_band
    summarize_experiment_periods
    print_period_takeaway

Eigenvalue solver
    hm76_constants
    default_hm76_background
    default_hm76_alpha
    compute_betae_for_eigen
    solve_hm76_eigenmodes

Eigenmode period utilities
    get_eigenmode_info
    eigenmode_period_table
    compare_period_with_eigen
    check_frozen_background
    check_frozen_background_many
    project_psi_time_onto_eigenmode
    period_from_complex_phase
    fft_peak_period
    diagnose_free_mode_period_from_run
    diagnose_free_mode_period_many
    reconstruct_stationary_plus_free_mode
"""

import numpy as np
import math

try:
    import scipy.linalg as la
except ImportError:
    la = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================
# Time integrators
# ============================================================

def run_hm76(hb, tau,
             s=2.0,
             dz=1000.0,
             imax=71,
             dt=360.0*15.0,
             n_days=150,
             alpha_on=False,
             wave_mean_feedback=True,
             mean_flow_coupling=1.0,
             verbose=False):
    """
    Run the Holton & Mass (1976) style 1D model for given bottom forcing.

    Parameters
    ----------
    hb : float
        Base-level forcing amplitude (m), appears in psi(0, t).
    tau : float
        Spin-up time scale for forcing (seconds), i.e. exp(-t/tau).
    s : float
        Zonal wavenumber index (default 2.0).
    dz : float
        Vertical grid spacing (m).
    imax : int
        Number of vertical grid points.
    dt : float
        Time step (seconds).
    n_days : int
        Total integration length in days.
    alpha_on : bool
        If True, use the original HM76 alpha(z) vertical profile; if False, alpha=0.
    wave_mean_feedback : bool
        If True, eddy wave flux updates the mean flow (full HM76).
        If False, wave sees a frozen initial mean flow (nowmfi control).
    mean_flow_coupling : float
        Multiplier on the wave-induced mean-flow tendency (default 1.0).
    verbose : bool
        If True, print progress.

    Returns
    -------
    out : dict
        z, ud, epz, tusu, psi_time, q_time, betae_time, qy_time,
        ubar/betae/wa/ua/epz at 25/30/35/40 km, hb, tau, dt, dz,
        k, l, f0, beta, eps, wave_mean_feedback, mean_flow_coupling,
        u_bg, betae_bg.
    """
    mmax = n_days * int(3600*24.0/dt)
    idd  = int((mmax - 1) / int(3600*24.0/dt))
    a = 6378000.0
    k = s/(a*np.cos(np.pi/3.0))
    l = 3.0/a
    omega_e = 7.29e-5
    g = 9.81
    f0 = 2.0*omega_e*np.sin(np.pi/3.0)
    beta = 2.0*omega_e*np.cos(np.pi/3.0)/a
    ensq = 4.0e-4
    h0 = 7000.0
    urz = 3.0e-3
    eps = 8.0/(3.0*np.pi)
    lrmsq = (f0*f0)/(4.0*ensq*h0*h0)
    r = (ensq*dz*dz)/(f0*f0)
    p = 2.0 + r*(k*k + l*l + lrmsq)
    pu = 2.0 + r*l*l
    vp = 1.0 + 0.5*dz/h0
    vm = 1.0 - 0.5*dz/h0
    tau = round(tau, 2)
    hb  = round(hb,  2)
    if verbose:
        tag = "WMFI on" if wave_mean_feedback else "WMFI off / frozen U"
        print('tau=', round(tau/86400.0, 3), 'days; hb=', hb, ';', tag,
              '; mean_flow_coupling=', mean_flow_coupling)

    # allocate arrays
    z     = np.zeros(imax)
    alpha = np.zeros(imax)
    u0    = np.zeros((imax, 4))
    ur    = np.zeros(imax)
    qy    = np.zeros((imax, 4))
    psi   = np.zeros((imax, 4), dtype=np.complex64)
    q     = np.zeros((imax, 4), dtype=np.complex64)
    ak = np.zeros(imax);  bk = np.zeros(imax, dtype=np.complex64)
    ck = np.zeros(imax);  dk = np.zeros(imax)
    ud        = np.zeros((imax, idd+1))
    epz       = np.zeros((imax, idd+1))
    tusu      = np.zeros((imax, idd+1))
    psi_time  = np.zeros((imax, idd+1), dtype=np.complex64)
    q_time    = np.zeros((imax, idd+1), dtype=np.complex64)
    qy_time   = np.zeros((imax, idd+1))
    betae_time = np.zeros((imax, idd+1))
    def _diag():
        a = np.zeros(idd+1); a[:] = np.nan; return a
    ubar25=_diag(); betae25=_diag(); wa25=_diag(); ua25=_diag(); epz25=_diag()
    ubar30=_diag(); betae30=_diag(); wa30=_diag(); ua30=_diag(); epz30=_diag()
    ubar35=_diag(); betae35=_diag(); wa35=_diag(); ua35=_diag(); epz35=_diag()
    ubar40=_diag(); betae40=_diag(); wa40=_diag(); ua40=_diag(); epz40=_diag()
    ur[:] = np.nan
    # z, alpha, ur
    for i in range(imax):
        z[i] = dz*i + 10000.0
        if alpha_on:
            alpha[i] = (1.5 + np.tanh((z[i]-35000.0)/h0))*1.0e-6
        ur[i] = 12.0 + 0.003*(z[i]-10000.0)
    # initial u0, qy
    for i in range(imax):
        u0[i, :] = 12.0 + (z[i]-10000.0)*(52.0-12.0)/40000.0
    for i in range(1, imax-1):
        qy[i, :] = -l*l*u0[i,0] + (f0*f0/ensq)*(
            (u0[i+1,0]+u0[i-1,0]-2.0*u0[i,0])/(dz*dz)
            - (u0[i+1,0]-u0[i-1,0])/(2.0*dz*h0))
    u_bg    = u0[:, 0].copy()
    qy_bg   = qy[:, 0].copy()
    betae_bg = beta - eps * qy_bg
    m = 0; mm = 0
    i25 = i30 = i35 = i40 = 0
    steps_per_day = int(3600*24.0/dt)

    while m < mmax:
        t = dt * m
        if not wave_mean_feedback:
            u0[:, 0]=u_bg; u0[:, 1]=u_bg; u0[:, 2]=u_bg
            qy[:, 0]=qy_bg; qy[:, 1]=qy_bg; qy[:, 2]=qy_bg
        # wave tendency (AB3)
        for i in range(1, imax-1):
            if wave_mean_feedback:
                betae1=beta-eps*qy[i,0]; betae2=beta-eps*qy[i,1]; betae3=beta-eps*qy[i,2]
                uadv1=-(0+1j)*k*eps*u0[i,0]*q[i,0]
                uadv2=-(0+1j)*k*eps*u0[i,1]*q[i,1]
                uadv3=-(0+1j)*k*eps*u0[i,2]*q[i,2]
            else:
                betae1=betae2=betae3=betae_bg[i]
                uadv1=-(0+1j)*k*eps*u_bg[i]*q[i,0]
                uadv2=-(0+1j)*k*eps*u_bg[i]*q[i,1]
                uadv3=-(0+1j)*k*eps*u_bg[i]*q[i,2]
            vadv1=-betae1*(0+1j)*k*psi[i,0]
            vadv2=-betae2*(0+1j)*k*psi[i,1]
            vadv3=-betae3*(0+1j)*k*psi[i,2]
            damp0=(f0*f0/ensq)*(0.5/h0)*alpha[i]*((psi[i+1,2]-psi[i-1,2])/(2.0*dz)+psi[i,2]/(2.0*h0))
            damp1=-(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2= (f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3=-(f0*f0/(ensq*dz*h0))*0.125*((alpha[i+1]+alpha[i])*psi[i+1,2]
                   -(alpha[i-1]+alpha[i])*psi[i-1,2]+(alpha[i+1]-alpha[i-1])*psi[i,2])
            q[i,3]=q[i,2]+(dt/12.0)*(5.0*uadv1-16.0*uadv2+23.0*uadv3
                                      +5.0*vadv1-16.0*vadv2+23.0*vadv3)
            q[i,3]+=dt*(damp0+damp1+damp2+damp3)
            zm=0.5*(z[i]+z[i-1])/h0; zp=0.5*(z[i]+z[i+1])/h0
            udamp1=-(f0*f0/ensq)*np.exp(z[i]/h0)*(1.0/dz)*(0.5*(alpha[i+1]+alpha[i])*np.exp(-zp)*((u0[i+1,2]-u0[i,2])/dz-urz))
            udamp2= (f0*f0/ensq)*np.exp(z[i]/h0)*(1.0/dz)*(0.5*(alpha[i-1]+alpha[i])*np.exp(-zm)*((u0[i,2]-u0[i-1,2])/dz-urz))
            flux1=0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(psi[i,0]*np.conj(psi[i+1,0]+psi[i-1,0]-2.0*psi[i,0])/(dz*dz))
            flux2=0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(psi[i,1]*np.conj(psi[i+1,1]+psi[i-1,1]-2.0*psi[i,1])/(dz*dz))
            flux3=0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(psi[i,2]*np.conj(psi[i+1,2]+psi[i-1,2]-2.0*psi[i,2])/(dz*dz))
            if wave_mean_feedback:
                qy[i,3]=qy[i,2]+(dt/12.0)*mean_flow_coupling*(5.0*flux1-16.0*flux2+23.0*flux3)+dt*(udamp1+udamp2)
            else:
                qy[i,3]=qy_bg[i]
        # invert psi from q
        ak[imax-2]=0.0; bk[imax-2]=0.0+0.0j
        for i in range(imax-2, -1, -1):
            ak[i-1]=-1.0/(ak[i]-p)
            bk[i-1]=(r*q[i,3]-bk[i])/(ak[i]-p)
        psi[0,3]=hb*(g/f0)*(1.0-np.exp(-t/tau))
        for i in range(imax-1):
            psi[i+1,3]=ak[i]*psi[i,3]+bk[i]
        # invert u0 from qy
        ck[imax-2]=1.0; dk[imax-2]=0.0
        for i in range(imax-2, -1, -1):
            ck[i-1]=-vp/(ck[i]*vm-pu)
            dk[i-1]=(r*qy[i,3]-dk[i]*vm)/(ck[i]*vm-pu)
        u0[0,3]=12.0
        for i in range(imax-1):
            u0[i+1,3]=ck[i]*u0[i,3]+dk[i]
        if not wave_mean_feedback:
            u0[:,3]=u_bg; qy[:,3]=qy_bg
        # shift time levels
        qy[:,0]=qy[:,1]; qy[:,1]=qy[:,2]; qy[:,2]=qy[:,3]
        q[:,0]=q[:,1];   q[:,1]=q[:,2];   q[:,2]=q[:,3]
        psi[:,0]=psi[:,1]; psi[:,1]=psi[:,2]; psi[:,2]=psi[:,3]
        u0[:,0]=u0[:,1]; u0[:,1]=u0[:,2]; u0[:,2]=u0[:,3]

        # daily output
        if m % steps_per_day == 0:
            ud[:,mm]=u0[:,2]; psi_time[:,mm]=psi[:,2]; q_time[:,mm]=q[:,2]
            qy_time[:,mm]=qy[:,2]; betae_time[:,mm]=beta-eps*qy[:,2]
            for i in range(1, imax-1):
                epz[i,mm]=-0.5*k*eps*np.imag(psi[i,1]*np.conj(psi[i+1,1]-psi[i-1,1])/(2.0*dz))
                tusu[i,mm]=eps*ud[i,mm]/(beta-eps*qy[i,1])-1.0/(k**2+l**2+lrmsq)
            def _lev(idx, diag, mm, flag):
                if flag != 0:
                    return flag
                diag[0][mm]=u0[idx,1]; diag[1][mm]=beta-eps*qy[idx,1]
                aa=abs(q[idx,1]); diag[2][mm]=eps*aa*aa*np.exp(z[idx]/7000.0)*0.25/diag[1][mm]
                diag[3][mm]=-0.5*k*eps*np.imag(psi[idx,1]*np.conj(psi[idx+1,1]-psi[idx-1,1])/(2.0*dz))
                diag[4][mm]=u0[idx,1]
                return mm if diag[1][mm] < 0 else 0
            i25=_lev(16,[ubar25,betae25,wa25,epz25,ua25],mm,i25)
            i30=_lev(21,[ubar30,betae30,wa30,epz30,ua30],mm,i30)
            i35=_lev(26,[ubar35,betae35,wa35,epz35,ua35],mm,i35)
            i40=_lev(31,[ubar40,betae40,wa40,epz40,ua40],mm,i40)
            mm += 1
        m += 1
    # scale EPZ
    c_epz = f0*f0/(ensq*4.0e-3)
    for i in range(1, imax-1):
        epz[i,:]=0.25*c_epz*np.exp(z[i]/7000.0)*epz[i,:]/(ud[i,20]*ud[i,20])
    return dict(
        z=z, ud=ud, epz=epz, tusu=tusu,
        psi_time=psi_time, q_time=q_time, qy_time=qy_time, betae_time=betae_time,
        ubar25=ubar25, betae25=betae25, wa25=wa25, ua25=ua25, epz25=epz25,
        ubar30=ubar30, betae30=betae30, wa30=wa30, ua30=ua30, epz30=epz30,
        ubar35=ubar35, betae35=betae35, wa35=wa35, ua35=ua35, epz35=epz35,
        ubar40=ubar40, betae40=betae40, wa40=wa40, ua40=ua40, epz40=epz40,
        hb=hb, tau=tau, dt=dt, dz=dz, k=k, l=l, f0=f0, beta=beta, eps=eps,
        wave_mean_feedback=wave_mean_feedback, mean_flow_coupling=mean_flow_coupling,
        u_bg=u_bg, betae_bg=betae_bg,
    )


def run_hm76_stationary(hb, tau,
                        s=2.0, dz=1000.0, imax=71,
                        dt=360.0*15.0, n_days=300,
                        alpha_on=True, verbose=False):
    """
    Linear HM76 run to obtain the stationary forced mode.
    Wave-mean-flow interaction OFF; background wind frozen.

    Returns
    -------
    out : dict
        z, ud (background repeated), epz, tusu, psi_time, q_time,
        psi_stationary (final time level), u_bg, hb, tau, dt, dz.
    """
    mmax = n_days * int(3600*24.0/dt)
    idd  = int((mmax-1)/int(3600*24.0/dt))
    a=6378000.0; k=s/(a*np.cos(np.pi/3.0)); l=3.0/a
    omega_e=7.29e-5; g=9.81
    f0=2.0*omega_e*np.sin(np.pi/3.0); beta=2.0*omega_e*np.cos(np.pi/3.0)/a
    ensq=4.0e-4; h0=7000.0; eps=8.0/(3.0*np.pi)
    lrmsq=(f0*f0)/(4.0*ensq*h0*h0); r=(ensq*dz*dz)/(f0*f0)
    p=2.0+r*(k*k+l*l+lrmsq)
    tau=round(tau,2); hb=round(hb,2)
    z=np.zeros(imax); alpha=np.zeros(imax)
    u0=np.zeros((imax,4)); qy=np.zeros((imax,4))
    psi=np.zeros((imax,4),dtype=np.complex64); q=np.zeros((imax,4),dtype=np.complex64)
    ak=np.zeros(imax); bk=np.zeros(imax,dtype=np.complex64)
    ud=np.zeros((imax,idd+1)); epz=np.zeros((imax,idd+1)); tusu=np.zeros((imax,idd+1))
    psi_time=np.zeros((imax,idd+1),dtype=np.complex64); q_time=np.zeros((imax,idd+1),dtype=np.complex64)
    def _d(): a=np.zeros(idd+1); a[:]=np.nan; return a
    ubar25=_d();betae25=_d();wa25=_d();ua25=_d();epz25=_d()
    ubar30=_d();betae30=_d();wa30=_d();ua30=_d();epz30=_d()
    ubar35=_d();betae35=_d();wa35=_d();ua35=_d();epz35=_d()
    ubar40=_d();betae40=_d();wa40=_d();ua40=_d();epz40=_d()
    for i in range(imax):
        z[i]=dz*i+10000.0
        alpha[i]=(1.5+np.tanh((z[i]-35000.0)/h0))*1.0e-6 if alpha_on else 0.0
    u_bg=np.array([12.0+(z[i]-10000.0)*(52.0-12.0)/40000.0 for i in range(imax)])
    for i in range(imax): u0[i,:]=u_bg[i]
    qy_bg=np.zeros(imax)
    for i in range(1,imax-1):
        qy_bg[i]=-l*l*u_bg[i]+(f0*f0/ensq)*((u_bg[i+1]+u_bg[i-1]-2.0*u_bg[i])/(dz*dz)
                                              -(u_bg[i+1]-u_bg[i-1])/(2.0*dz*h0))
        qy[i,:]=qy_bg[i]
    m=0;mm=0;i25=i30=i35=i40=0
    steps_per_day=int(3600*24.0/dt)
    while m < mmax:
        t=dt*m
        u0[:,0]=u_bg; u0[:,1]=u_bg; u0[:,2]=u_bg
        qy[:,0]=qy_bg; qy[:,1]=qy_bg; qy[:,2]=qy_bg
        for i in range(1,imax-1):
            betae1=beta-eps*qy[i,0]; betae2=beta-eps*qy[i,1]; betae3=beta-eps*qy[i,2]
            uadv1=-(0+1j)*k*eps*u0[i,0]*q[i,0]; uadv2=-(0+1j)*k*eps*u0[i,1]*q[i,1]; uadv3=-(0+1j)*k*eps*u0[i,2]*q[i,2]
            vadv1=-betae1*(0+1j)*k*psi[i,0]; vadv2=-betae2*(0+1j)*k*psi[i,1]; vadv3=-betae3*(0+1j)*k*psi[i,2]
            damp0=(f0*f0/ensq)*(0.5/h0)*alpha[i]*((psi[i+1,2]-psi[i-1,2])/(2.0*dz)+psi[i,2]/(2.0*h0))
            damp1=-(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2= (f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3=-(f0*f0/(ensq*dz*h0))*0.125*((alpha[i+1]+alpha[i])*psi[i+1,2]
                   -(alpha[i-1]+alpha[i])*psi[i-1,2]+(alpha[i+1]-alpha[i-1])*psi[i,2])
            q[i,3]=q[i,2]+(dt/12.0)*(5.0*uadv1-16.0*uadv2+23.0*uadv3+5.0*vadv1-16.0*vadv2+23.0*vadv3)
            q[i,3]+=dt*(damp0+damp1+damp2+damp3)
        ak[imax-2]=0.0; bk[imax-2]=0.0+0.0j
        for i in range(imax-2,-1,-1):
            ak[i-1]=-1.0/(ak[i]-p); bk[i-1]=(r*q[i,3]-bk[i])/(ak[i]-p)
        psi[0,3]=hb*(g/f0)*(1.0-np.exp(-t/tau))
        for i in range(imax-1): psi[i+1,3]=ak[i]*psi[i,3]+bk[i]
        q[:,0]=q[:,1]; q[:,1]=q[:,2]; q[:,2]=q[:,3]
        psi[:,0]=psi[:,1]; psi[:,1]=psi[:,2]; psi[:,2]=psi[:,3]
        if m % steps_per_day == 0:
            ud[:,mm]=u_bg[:]; psi_time[:,mm]=psi[:,2]; q_time[:,mm]=q[:,2]
            for i in range(1,imax-1):
                epz[i,mm]=-0.5*k*eps*np.imag(psi[i,1]*np.conj(psi[i+1,1]-psi[i-1,1])/(2.0*dz))
                tusu[i,mm]=eps*ud[i,mm]/(beta-eps*qy[i,1])-1.0/(k**2+l**2+lrmsq)
            mm+=1
        m+=1
    c_epz=f0*f0/(ensq*4.0e-3)
    for i in range(1,imax-1):
        epz[i,:]=0.25*c_epz*np.exp(z[i]/7000.0)*epz[i,:]/(ud[i,20]*ud[i,20])
    return dict(
        z=z, ud=ud, epz=epz, tusu=tusu, psi_time=psi_time, q_time=q_time,
        psi_stationary=psi[:,2].copy(), u_bg=u_bg,
        hb=hb, tau=tau, dt=dt, dz=dz,
    )


def run_hm76_transient(s=2.0, dz=1000.0, imax=71, dt=360.0*15.0,
                       n_days=300, alpha_on=True, verbose=False,
                       z0=30000.0, sigma=10000.0, amp=1.0):
    """
    Linear HM76 run to obtain the least-damped transient (traveling) mode.
    No bottom forcing; starts from a localized Gaussian perturbation in q.

    Returns
    -------
    out : dict
        z, ud, psi_last, psi_time, q_time, u_bg, qy_bg, k, l, f0, beta, dt, dz.
    """
    mmax=n_days*int(3600*24.0/dt); idd=int((mmax-1)/int(3600*24.0/dt))
    a=6378000.0; k=s/(a*np.cos(np.pi/3.0)); l=3.0/a
    omega_e=7.29e-5; f0=2.0*omega_e*np.sin(np.pi/3.0); beta=2.0*omega_e*np.cos(np.pi/3.0)/a
    ensq=4.0e-4; h0=7000.0; eps=8.0/(3.0*np.pi)
    lrmsq=(f0*f0)/(4.0*ensq*h0*h0); r=(ensq*dz*dz)/(f0*f0); p=2.0+r*(k*k+l*l+lrmsq)
    imax=int(imax)
    z=np.zeros(imax); alpha=np.zeros(imax)
    u0=np.zeros((imax,4)); qy=np.zeros((imax,4))
    psi=np.zeros((imax,4),dtype=np.complex64); q=np.zeros((imax,4),dtype=np.complex64)
    ak=np.zeros(imax); bk=np.zeros(imax,dtype=np.complex64)
    ud=np.zeros((imax,idd+1)); psi_time=np.zeros((imax,idd+1),dtype=np.complex64)
    q_time=np.zeros((imax,idd+1),dtype=np.complex64)
    for i in range(imax):
        z[i]=dz*i+10000.0
        alpha[i]=(1.5+np.tanh((z[i]-35000.0)/h0))*1.0e-6 if alpha_on else 0.0
    u_bg=np.array([12.0+(z[i]-10000.0)*(52.0-12.0)/40000.0 for i in range(imax)])
    for i in range(imax): u0[i,:]=u_bg[i]
    qy_bg=np.zeros(imax)
    for i in range(1,imax-1):
        qy_bg[i]=-l*l*u_bg[i]+(f0*f0/ensq)*((u_bg[i+1]+u_bg[i-1]-2.0*u_bg[i])/(dz*dz)
                                              -(u_bg[i+1]-u_bg[i-1])/(2.0*dz*h0))
        qy[i,:]=qy_bg[i]
        q_init=amp*np.exp(-0.5*((z[i]-z0)/sigma)**2)
        q[i,0]=q[i,1]=q[i,2]=q_init+0.0j
    # invert initial psi
    ak[imax-2]=0.0; bk[imax-2]=0.0+0.0j
    for i in range(imax-2,-1,-1):
        ak[i-1]=-1.0/(ak[i]-p); bk[i-1]=(r*q[i,2]-bk[i])/(ak[i]-p)
    psi[0,2]=0.0+0.0j
    for i in range(imax-1): psi[i+1,2]=ak[i]*psi[i,2]+bk[i]
    psi[:,0]=psi[:,1]=psi[:,2]
    m=0; mm=0; steps_per_day=int(3600*24.0/dt)
    while m < mmax:
        t=dt*m
        u0[:,0]=u0[:,1]=u0[:,2]=u_bg
        qy[:,0]=qy[:,1]=qy[:,2]=qy_bg
        for i in range(1,imax-1):
            betae1=beta-eps*qy[i,0]; betae2=beta-eps*qy[i,1]; betae3=beta-eps*qy[i,2]
            uadv1=-(0+1j)*k*eps*u0[i,0]*q[i,0]; uadv2=-(0+1j)*k*eps*u0[i,1]*q[i,1]; uadv3=-(0+1j)*k*eps*u0[i,2]*q[i,2]
            vadv1=-betae1*(0+1j)*k*psi[i,0]; vadv2=-betae2*(0+1j)*k*psi[i,1]; vadv3=-betae3*(0+1j)*k*psi[i,2]
            damp0=(f0*f0/ensq)*(0.5/h0)*alpha[i]*((psi[i+1,2]-psi[i-1,2])/(2.0*dz)+psi[i,2]/(2.0*h0))
            damp1=-(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2= (f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3=-(f0*f0/(ensq*dz*h0))*0.125*((alpha[i+1]+alpha[i])*psi[i+1,2]
                   -(alpha[i-1]+alpha[i])*psi[i-1,2]+(alpha[i+1]-alpha[i-1])*psi[i,2])
            q[i,3]=q[i,2]+(dt/12.0)*(5.0*uadv1-16.0*uadv2+23.0*uadv3+5.0*vadv1-16.0*vadv2+23.0*vadv3)
            q[i,3]+=dt*(damp0+damp1+damp2+damp3)
        ak[imax-2]=0.0; bk[imax-2]=0.0+0.0j
        for i in range(imax-2,-1,-1):
            ak[i-1]=-1.0/(ak[i]-p); bk[i-1]=(r*q[i,3]-bk[i])/(ak[i]-p)
        psi[0,3]=0.0+0.0j
        for i in range(imax-1): psi[i+1,3]=ak[i]*psi[i,3]+bk[i]
        q[:,0]=q[:,1]; q[:,1]=q[:,2]; q[:,2]=q[:,3]
        psi[:,0]=psi[:,1]; psi[:,1]=psi[:,2]; psi[:,2]=psi[:,3]
        if m % steps_per_day == 0:
            ud[:,mm]=u_bg[:]; psi_time[:,mm]=psi[:,2]; q_time[:,mm]=q[:,2]; mm+=1
        m+=1
    return dict(
        z=z, ud=ud, psi_last=psi[:,2].copy(), psi_time=psi_time, q_time=q_time,
        u_bg=u_bg, qy_bg=qy_bg, k=k, l=l, f0=f0, beta=beta, dt=dt, dz=dz,
    )


def extract_transient_mode(psi_time, z, k,
                           dt_out_days=1.0, n_fit=None, level_ref=None,
                           verbose=False, q_time=None):
    """
    Extract the least-damped transient mode from psi_time(z, t) via
    phase-aligned time averaging.

    Returns
    -------
    out : dict
        psi_mode, q_mode (or None), c_phase, omega, level_ref, t_fit, phase_fit.
    """
    psi_time=np.asarray(psi_time); z=np.asarray(z)
    nz,nt=psi_time.shape
    if n_fit is None: n_fit=max(nt//3,4)
    i_start=nt-n_fit
    psi_win=psi_time[:,i_start:]
    q_win=np.asarray(q_time)[:,i_start:] if q_time is not None else None
    if level_ref is None:
        level_ref=np.argmax(np.mean(np.abs(psi_win),axis=1))
    dt_out_sec=dt_out_days*86400.0
    t_fit=np.arange(i_start,nt)*dt_out_sec
    psi_ref=psi_time[level_ref,i_start:]
    phase_unwrap=np.unwrap(np.angle(psi_ref))
    a,b=np.polyfit(t_fit,phase_unwrap,1)
    omega=-a; c_phase=omega/k
    psi_aligned=np.zeros_like(psi_win,dtype=np.complex128)
    q_aligned=np.zeros_like(q_win,dtype=np.complex128) if q_win is not None else None
    for j in range(n_fit):
        rot=np.exp(-1j*np.angle(psi_win[level_ref,j]))
        psi_aligned[:,j]=psi_win[:,j]*rot
        if q_aligned is not None: q_aligned[:,j]=q_win[:,j]*rot
    psi_mode_raw=np.mean(psi_aligned,axis=1)
    q_mode_raw=np.mean(q_aligned,axis=1) if q_aligned is not None else None
    mx=np.max(np.abs(psi_mode_raw))
    if mx>0:
        psi_mode=psi_mode_raw/mx
        q_mode=q_mode_raw/mx if q_mode_raw is not None else None
    else:
        psi_mode=psi_mode_raw; q_mode=q_mode_raw
    return dict(psi_mode=psi_mode, q_mode=q_mode, c_phase=c_phase, omega=omega,
                level_ref=level_ref, t_fit=t_fit, phase_fit=phase_unwrap)


# ============================================================
# Diagnostics
# ============================================================

def compute_m2(out, c=0.0, s=2.0):
    """
    Compute refractive index squared m²(z,t) from HM76 output.

    Parameters
    ----------
    out : dict   Output from run_hm76().
    c   : float  Phase speed in m/s (c=0 for stationary wave).
    s   : float  Zonal wavenumber index.

    Returns
    -------
    m2 : ndarray, shape (nz, nt)
    """
    a=6378000.0; omega_e=7.29e-5; ensq=4.0e-4; h0=7000.0
    k=s/(a*np.cos(np.pi/3.0)); l=3.0/a
    f0=2.0*omega_e*np.sin(np.pi/3.0)
    ld_inv2=f0**2/(4.0*ensq*h0**2)
    K2=k**2+l**2+ld_inv2
    ud=out["ud"]; betae=out["betae_time"]; eps=out["eps"]
    denom=eps*ud-c
    m2=(ensq/f0**2)*(betae/denom-K2)
    m2[np.abs(denom)<1e-6]=np.nan
    return m2


def compute_Ucrit_fields(betae_time, c_phase, *, k, l, ensq, h0, f0, eps):
    """
    Compute critical-level wind fields Ucrit_stat and Ucrit_tran.

    Parameters
    ----------
    betae_time : ndarray, shape (nz, nt)
    c_phase    : float   Transient wave phase speed (m/s).

    Returns
    -------
    Ucrit_stat, Ucrit_tran : ndarray, shape (nz, nt)
    """
    N=np.sqrt(ensq); LD=2.0*N*h0/f0
    D=k**2+l**2+1.0/(LD**2)
    Ucrit_stat=betae_time/D/eps
    Ucrit_tran=c_phase+betae_time/D/eps
    return Ucrit_stat, Ucrit_tran


def compute_epflux_zt(out, density_weighted=True):
    """
    Compute vertical EP flux F_z(z,t) from HM76 output (psi_time).

    This is the raw (unscaled) vertical component of the EP flux,
    proportional to the vertical eddy heat flux in QG:
        F_z ∝ Im(ψ * conj(dψ/dz))

    Parameters
    ----------
    out : dict
        Output from run_hm76(). Must contain psi_time, z, k, eps, dz.
    density_weighted : bool
        If True, multiply by exp(z/H) to get the density-weighted flux
        (conserved in the absence of wave-mean flow interaction).
        If False, return the local flux amplitude.

    Returns
    -------
    Fz : ndarray, shape (nz, nt)
        Vertical EP flux. Positive = upward wave activity propagation.
        Units depend on density_weighted flag.
    """
    psi = out["psi_time"]  # complex, shape (nz, nt)
    z = out["z"]
    k = out["k"]
    eps = out["eps"]
    dz = out["dz"]
    h0 = 7000.0
    nz, nt = psi.shape

    # F_z[i] = -0.5 * k * eps * Im(psi[i] * conj(psi[i+1] - psi[i-1]) / (2*dz))
    Fz = np.zeros((nz, nt), dtype=float)
    for i in range(1, nz - 1):
        dpsi_dz = (psi[i + 1, :] - psi[i - 1, :]) / (2.0 * dz)
        Fz[i, :] = -0.5 * k * eps * np.imag(psi[i, :] * np.conj(dpsi_dz))

    if density_weighted:
        # Multiply by exp(z/H) so that div(F) = wave forcing
        rho_factor = np.exp(z / h0)
        Fz = Fz * rho_factor[:, np.newaxis]

    return Fz


def compute_turning_levels(m2, z):
    """
    Find m²=0 turning-level heights as a function of time.

    For each time step, finds all heights where m² crosses zero
    (by linear interpolation between adjacent grid points).

    Parameters
    ----------
    m2 : ndarray, shape (nz, nt)
        Refractive index squared from compute_m2().
    z  : ndarray, shape (nz,)
        Height array in meters.

    Returns
    -------
    turning_levels : list of lists
        turning_levels[t] is a list of heights (m) where m²=0 at time t.
    lowest_turning : ndarray, shape (nt,)
        The lowest turning level at each time (m). NaN if no turning level.
    """
    nz, nt = m2.shape
    turning_levels = []
    lowest_turning = np.full(nt, np.nan)

    for t in range(nt):
        crossings = []
        for i in range(nz - 1):
            v0 = m2[i, t]
            v1 = m2[i + 1, t]
            if np.isnan(v0) or np.isnan(v1):
                continue
            if v0 * v1 < 0:
                # Linear interpolation for the crossing height
                z_cross = z[i] + (z[i + 1] - z[i]) * (-v0) / (v1 - v0)
                crossings.append(z_cross)
        turning_levels.append(crossings)
        if crossings:
            lowest_turning[t] = min(crossings)

    return turning_levels, lowest_turning


def compute_min_m2(m2, z, z1_km=15.0, z2_km=50.0):
    """
    Scalar propagation-barrier diagnostic: M(t) = min_{z∈[z1,z2]} m²(z,t).

    Parameters
    ----------
    m2 : ndarray, shape (nz, nt)
        Refractive index squared from compute_m2().
    z  : ndarray, shape (nz,)
        Height array in meters.
    z1_km, z2_km : float
        Vertical range in km for the minimum search.

    Returns
    -------
    M : ndarray, shape (nt,)
        Minimum m² over the specified height range at each time step.
    """
    z_km = z / 1000.0
    mask = (z_km >= z1_km) & (z_km <= z2_km)
    m2_region = m2[mask, :]
    # Replace NaN with +inf so they don't contaminate the min
    m2_safe = np.where(np.isnan(m2_region), np.inf, m2_region)
    M = np.min(m2_safe, axis=0)
    M[np.isinf(M)] = np.nan
    return M


# ============================================================
# Period estimation utilities
# ============================================================

def estimate_period_days_fft_band(x, dt_days=1.0,
                                  day_min=20.0, day_max=80.0,
                                  detrend=True, remove_mean=True,
                                  plot_spectrum=False, title=None):
    """
    Estimate the dominant period in a prescribed band using FFT.

    Returns
    -------
    T_days : float   Dominant period (days); nan if no peak found.
    f0     : float   Corresponding frequency (1/day).
    """
    import matplotlib.pyplot as plt
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)<5: return np.nan, np.nan
    t=np.arange(len(x))*dt_days
    if detrend:
        a,b=np.polyfit(t,x,1); x=x-(a*t+b)
    if remove_mean: x=x-np.mean(x)
    n=len(x); freqs=np.fft.rfftfreq(n,d=dt_days); spec=np.abs(np.fft.rfft(x))**2
    fmin=1.0/day_max; fmax=1.0/day_min
    mask=(freqs>=fmin)&(freqs<=fmax)&(freqs>0)
    if not np.any(mask): return np.nan, np.nan
    f0=freqs[mask][np.argmax(spec[mask])]; T_days=1.0/f0
    if plot_spectrum:
        valid=freqs>0; periods=1.0/freqs[valid]; spec_v=spec[valid]
        plt.figure(figsize=(6,4)); plt.plot(periods,spec_v)
        plt.axvline(T_days,linestyle="--",label=f"T={T_days:.1f} d")
        plt.axvspan(day_min,day_max,alpha=0.15,label="search band")
        plt.xlim(0,max(120,day_max*1.2)); plt.xlabel("Period (days)"); plt.ylabel("Power")
        plt.title(title or "FFT spectrum"); plt.grid(True,linestyle=":"); plt.legend()
        plt.tight_layout(); plt.show()
    return T_days, f0

# Backward-compatible alias.
estimate_period_days_fft = estimate_period_days_fft_band


def estimate_c_from_series_band(x, k, dt_days=1.0, day_min=20.0, day_max=80.0,
                                i1=None, i2=None, detrend=True,
                                plot_spectrum=False, title=None):
    """Estimate phase speed from a time series using FFT period."""
    x=np.asarray(x,dtype=float)
    if i1 is not None or i2 is not None: x=x[i1:i2]
    T_days,_=estimate_period_days_fft_band(x,dt_days=dt_days,day_min=day_min,day_max=day_max,
                                           detrend=detrend,plot_spectrum=plot_spectrum,title=title)
    if not np.isfinite(T_days): return np.nan, T_days
    c=2.0*np.pi/(k*T_days*86400.0)
    return c, T_days


def summarize_experiment_periods(experiments, diagnostics=("A_res","psi_amp30","psi_amp40","u10"),
                                  i1=20, i2=200, day_min=20.0, day_max=80.0,
                                  dt_days=1.0, plot_spectra=False):
    """
    Estimate dominant periods for all experiments across multiple diagnostics.

    Returns
    -------
    DataFrame or list of dicts.
    """
    rows=[]
    for key in sorted(experiments):
        exp=experiments[key]; out=exp["out"]
        k_run=out.get("k", 2.0/(6378000.0*np.cos(np.pi/3.0)))
        for diag in diagnostics:
            y,_,_=get_experiment_series(exp, diagnostic=diag)
            c,T=estimate_c_from_series_band(y,k_run,dt_days=dt_days,day_min=day_min,day_max=day_max,
                                            i1=i1,i2=i2,detrend=True,plot_spectrum=plot_spectra,
                                            title=f"{key}: {diag}")
            rows.append(dict(key=key,tau_days=exp["tau_days"],feedback=exp["feedback_tag"],
                             diagnostic=diag,period_days=T,phase_speed_mps=c,
                             window=f"{i1}-{i2} d",search_band=f"{day_min}-{day_max} d"))
    if pd is not None: return pd.DataFrame(rows)
    return rows


def get_experiment_series(exp, diagnostic="A_res"):
    """
    Extract a 1D time series from an experiment dict by diagnostic name.

    Supported diagnostics: A_res, psi_amp30, psi_amp40, u10, u30.
    """
    out=exp["out"]
    if diagnostic=="A_res":
        wa=exp.get("WA",{})
        y=np.asarray(wa.get("A_res",out.get("wa30",np.zeros(out["ud"].shape[1]))),dtype=float)
        return y, "Wave activity residual A_res", "m²/s"
    elif diagnostic=="psi_amp30":
        idx=21; y=np.abs(out["psi_time"][idx,:])
        return y, "|psi| at 30 km", "m²/s"
    elif diagnostic=="psi_amp40":
        idx=31; y=np.abs(out["psi_time"][idx,:])
        return y, "|psi| at 40 km", "m²/s"
    elif diagnostic=="u10":
        idx=out["ud"].shape[0]//3; y=out["ud"][idx,:]
        return y, "u at ~10 hPa", "m/s"
    elif diagnostic=="u30":
        idx=21; y=out["ud"][idx,:]
        return y, "u at 30 km", "m/s"
    else:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")


def print_period_takeaway(period_table, diagnostic="A_res"):
    """Print nowmfi-vs-full period comparison."""
    if pd is not None:
        sub=period_table[period_table["diagnostic"]==diagnostic].copy()
        if sub.empty: print(f"No rows for diagnostic={diagnostic}"); return
        print(f"\nnowmfi-vs-full period comparison using {diagnostic}:")
        for tau in sorted(sub["tau_days"].unique()):
            n=sub[(sub["tau_days"]==tau)&(sub["feedback"]=="nowmfi")]
            f=sub[(sub["tau_days"]==tau)&(sub["feedback"]=="full")]
            Tn=n["period_days"].iloc[0] if len(n) else np.nan
            Tf=f["period_days"].iloc[0] if len(f) else np.nan
            print(f"  tau={tau:.2f} d: nowmfi={Tn:.2f} d, full={Tf:.2f} d, full/nowmfi={Tf/Tn:.2f}")
    else:
        print(period_table)


# ============================================================
# Eigenvalue solver
# ============================================================

def hm76_constants(s=2.0, dz=1000.0):
    """Return constants consistent with the HM76 time-stepping code."""
    a=6378000.0; phi0=np.pi/3.0
    k=s/(a*np.cos(phi0)); l=3.0/a
    omega_e=7.29e-5
    f0=2.0*omega_e*np.sin(phi0); beta=2.0*omega_e*np.cos(phi0)/a
    ensq=4.0e-4; h0=7000.0; eps=8.0/(3.0*np.pi)
    return dict(s=s,dz=dz,a=a,k=k,l=l,omega_earth=omega_e,f0=f0,beta=beta,
                ensq=ensq,h0=h0,eps=eps,
                f2_over_N2=f0**2/ensq, LD_inv2=f0**2/(4.0*ensq*h0**2))


def default_hm76_background(z):
    """Same initial background wind as the HM76 time-stepping code."""
    return 12.0 + (z - 10000.0) * (52.0 - 12.0) / 40000.0


def default_hm76_alpha(z, h0=7000.0, alpha_on=True):
    """Same radiative damping profile as the HM76 time-stepping code."""
    if alpha_on:
        return (1.5 + np.tanh((z - 35000.0) / h0)) * 1.0e-6
    return np.zeros_like(z)


def compute_betae_for_eigen(z, u_bg, const):
    """
    Compute beta_e'(z) for the eigenvalue problem:
        beta_e' = beta + eps*l²*U₀ - (f0²/N²)*eps*(Uzz - Uz/H)
    """
    dz=const["dz"]; beta=const["beta"]; eps=const["eps"]; l=const["l"]
    h0=const["h0"]; f2N2=const["f2_over_N2"]; imax=len(z)
    betae=np.zeros(imax,dtype=np.complex128)
    for i in range(1,imax-1):
        Uzz=(u_bg[i+1]+u_bg[i-1]-2.0*u_bg[i])/dz**2
        Uz=(u_bg[i+1]-u_bg[i-1])/(2.0*dz)
        betae[i]=beta+eps*l**2*u_bg[i]-f2N2*eps*(Uzz-Uz/h0)
    betae[0]=betae[1]; betae[-1]=betae[-2]
    return betae


def solve_hm76_eigenmodes(s=2.0, dz=1000.0, imax=71,
                          alpha_on=True, u_bg=None, n_print=10):
    """
    Solve the HM76 free-mode eigenvalue problem: M A = omega N A.

    Boundary conditions: A[0] = A[imax-1] = 0.

    Parameters
    ----------
    s        : float   Zonal wavenumber index.
    dz       : float   Vertical grid spacing (m).
    imax     : int     Number of vertical levels.
    alpha_on : bool    Include radiative damping.
    u_bg     : array or None   Background wind; None → default HM76 profile.
    n_print  : int     Number of leading modes to print.

    Returns
    -------
    out : dict
        z, s, k, l, omega, modes, order, node_count, roughness,
        top_fraction, interior_fraction, centroid_km, score,
        u_bg, alpha, betae, const, M, Nmat.
    """
    if la is None:
        raise ImportError("scipy.linalg is required for solve_hm76_eigenmodes.")
    const=hm76_constants(s=s,dz=dz)
    k=const["k"]; l=const["l"]; beta=const["beta"]; eps=const["eps"]
    h0=const["h0"]; f2N2=const["f2_over_N2"]; LD2=const["LD_inv2"]
    z=np.arange(imax)*dz+10000.0
    if u_bg is None:
        u_bg=default_hm76_background(z)
    else:
        u_bg=np.asarray(u_bg,dtype=float)
    alpha=default_hm76_alpha(z,h0=h0,alpha_on=alpha_on)
    betae=compute_betae_for_eigen(z,u_bg,const)
    n=imax-2
    M=np.zeros((n,n),dtype=np.complex128); Nmat=np.zeros((n,n),dtype=np.complex128)
    def add(mat,row,idx,val):
        if 1<=idx<=imax-2: mat[row,idx-1]+=val
    for i in range(1,imax-1):
        row=i-1
        Bm=f2N2/dz**2; B0=-(k**2+l**2+LD2)-2.0*f2N2/dz**2; Bp=f2N2/dz**2
        add(Nmat,row,i-1,Bm); add(Nmat,row,i,B0); add(Nmat,row,i+1,Bp)
        pref=eps*k*u_bg[i]
        add(M,row,i-1,pref*Bm); add(M,row,i,pref*B0+betae[i]*k); add(M,row,i+1,pref*Bp)
        if alpha_on:
            add(M,row,i-1,-1j*alpha[i]*f2N2/dz**2)
            add(M,row,i,  -1j*alpha[i]*(-2.0*f2N2/dz**2-LD2))
            add(M,row,i+1,-1j*alpha[i]*f2N2/dz**2)
            az=(alpha[i+1]-alpha[i-1])/(2.0*dz)
            C=-1j*f2N2*(az-alpha[i]/(2.0*h0))
            add(M,row,i-1,C*(-1.0/(2.0*dz))); add(M,row,i,C*(1.0/(2.0*h0))); add(M,row,i+1,C*(1.0/(2.0*dz)))
    omega,vecs=la.eig(M,Nmat)
    modes=np.zeros((imax,len(omega)),dtype=np.complex128); modes[1:-1,:]=vecs
    for j in range(len(omega)):
        amp=np.nanmax(np.abs(modes[:,j]))
        if amp>0: modes[:,j]/=amp
    # diagnostics for ranking
    node_count=np.zeros(len(omega),dtype=int); roughness=np.zeros(len(omega))
    top_fraction=np.zeros(len(omega)); interior_fraction=np.zeros(len(omega)); centroid_km=np.zeros(len(omega))
    z_km=z/1000.0; top_mask=z_km>=70.0; int_mask=(z_km>=15.0)&(z_km<=65.0)
    for j in range(len(omega)):
        A=modes[:,j]; idx=np.argmax(np.abs(A)); A_rot=A*np.exp(-1j*np.angle(A[idx]))
        Ar=np.real(A_rot[1:-1]); tol=1e-6*np.nanmax(np.abs(Ar)); Ar2=Ar.copy(); Ar2[np.abs(Ar2)<tol]=0.0
        signs=np.sign(Ar2); snz=signs[signs!=0]
        node_count[j]=np.sum(snz[1:]*snz[:-1]<0) if len(snz)>=2 else 0
        d2=A[2:]-2.0*A[1:-1]+A[:-2]; roughness[j]=np.sum(np.abs(d2)**2)
        en=np.abs(A)**2; tot=np.nansum(en)
        if tot>0:
            top_fraction[j]=np.nansum(en[top_mask])/tot
            interior_fraction[j]=np.nansum(en[int_mask])/tot
            centroid_km[j]=np.nansum(z_km*en)/tot
        else:
            top_fraction[j]=interior_fraction[j]=centroid_km[j]=np.nan
    rough_norm=roughness/np.nanmax(roughness) if np.nanmax(roughness)>0 else roughness
    damp_abs=np.abs(np.imag(omega))
    damp_norm=damp_abs/np.nanmax(damp_abs) if np.nanmax(damp_abs)>0 else damp_abs
    score_base=10.0*node_count+1.0*rough_norm+0.2*damp_norm
    valid=(np.isfinite(score_base)&(top_fraction<0.45)&(interior_fraction>0.25)&(centroid_km<65.0))
    vi=np.where(valid)[0]; ii=np.where(~valid)[0]
    order=np.concatenate([vi[np.argsort(score_base[vi])], ii[np.argsort(score_base[ii])]])
    score=score_base.copy(); score[~valid]+=1.0e6
    if n_print>0:
        print(""); print("="*105)
        print(f"HM76 eigenmodes: s={s:g}, alpha_on={alpha_on}")
        print("="*105)
        print("rank | mode_id | omega_r [1/s] | omega_i [1/s] | period [days] | c_phase [m/s] | nodes | top_frac | int_frac | z_cent")
        print("-----|---------|---------------|---------------|---------------|---------------|-------|----------|----------|--------")
        for rank,j in enumerate(order[:n_print]):
            wr=np.real(omega[j]); wi=np.imag(omega[j])
            T=2.0*np.pi/np.abs(wr)/86400.0 if np.abs(wr)>0 else np.inf
            c=wr/k
            print(f"{rank:4d} | {j:7d} | {wr: .4e} | {wi: .4e} | {T: .3f} | {c: .3f} | "
                  f"{node_count[j]:5d} | {top_fraction[j]:8.3f} | {interior_fraction[j]:8.3f} | {centroid_km[j]:6.1f}")
    return dict(z=z,s=s,k=k,l=l,omega=omega,modes=modes,order=order,
                node_count=node_count,roughness=roughness,
                top_fraction=top_fraction,interior_fraction=interior_fraction,centroid_km=centroid_km,
                score=score,u_bg=u_bg,alpha=alpha,betae=betae,const=const,M=M,Nmat=Nmat)


# ============================================================
# Eigenmode period utilities
# ============================================================

def get_eigenmode_info(eig_out, mode_rank=0):
    """Return eigenvalue-based period and phase speed for one selected eigenmode."""
    omega_all=eig_out["omega"]; modes=eig_out["modes"]; order=eig_out["order"]; k=eig_out["k"]
    j=int(order[mode_rank]); omega_m=omega_all[j]
    A=modes[:,j].astype(np.complex128).copy()
    idx=np.nanargmax(np.abs(A))
    if np.abs(A[idx])>0: A=A*np.exp(-1j*np.angle(A[idx]))
    wr=np.real(omega_m); wi=np.imag(omega_m)
    T=2.0*np.pi/np.abs(wr)/86400.0 if np.abs(wr)>0 else np.inf
    return dict(z=eig_out["z"],mode_rank=mode_rank,mode_id=j,omega=omega_m,
                omega_r=wr,omega_i=wi,period_days=T,c_phase=wr/k,mode=A)


def eigenmode_period_table(eig_out, n_modes=8):
    """Make a clean table of the leading eigenmodes."""
    rows=[]
    for rank in range(n_modes):
        info=get_eigenmode_info(eig_out,rank); j=info["mode_id"]
        rows.append(dict(rank=rank,mode_id=j,omega_r=info["omega_r"],omega_i=info["omega_i"],
                         period_days=info["period_days"],c_phase_mps=info["c_phase"],
                         node_count=eig_out.get("node_count",[np.nan]*len(eig_out["omega"]))[j],
                         top_fraction=eig_out.get("top_fraction",[np.nan]*len(eig_out["omega"]))[j],
                         interior_fraction=eig_out.get("interior_fraction",[np.nan]*len(eig_out["omega"]))[j],
                         centroid_km=eig_out.get("centroid_km",[np.nan]*len(eig_out["omega"]))[j],
                         score=eig_out.get("score",[np.nan]*len(eig_out["omega"]))[j]))
    return pd.DataFrame(rows) if pd is not None else rows


def compare_period_with_eigen(eig_out, target_period_days, n_compare=8,
                              target_name="target phase period"):
    """Compare a diagnosed period with eigenmode periods. Use phase-based periods only."""
    omega=eig_out["omega"]; order=eig_out["order"]; k=eig_out["k"]; s=eig_out["s"]
    print(""); print("="*110)
    print(f"Compare {target_name} with HM76 eigenmodes, s={s:g}")
    print(f"{target_name} = {target_period_days:.3f} days"); print("="*110)
    print("rank | mode_id | omega_r [1/s] | omega_i [1/s] | period [days] | c_phase [m/s] | |T-Ttarget| [days]")
    for rank,j in enumerate(order[:n_compare]):
        wr=np.real(omega[j]); wi=np.imag(omega[j])
        T=2.0*np.pi/np.abs(wr)/86400.0 if np.abs(wr)>0 else np.inf
        print(f"{rank:4d} | {j:7d} | {wr: .4e} | {wi: .4e} | {T: .3f} | {wr/k: .3f} | {abs(T-target_period_days): .3f}")


def check_frozen_background(out, key=None):
    """Check whether a nowmfi run really keeps u and beta_e fixed."""
    key=key or out.get("key","run")
    ud=out.get("ud"); u_bg=out.get("u_bg")
    betae_time=out.get("betae_time"); betae_bg=out.get("betae_bg")
    max_du=np.nanmax(np.abs(ud-u_bg[:,None])) if ud is not None and u_bg is not None else np.nan
    max_db=np.nanmax(np.abs(betae_time-betae_bg[:,None])) if betae_time is not None and betae_bg is not None else np.nan
    return dict(key=key,feedback=out.get("feedback","unknown"),
                wave_mean_feedback=out.get("wave_mean_feedback",np.nan),
                mean_flow_coupling=out.get("mean_flow_coupling",np.nan),
                max_abs_du=max_du,max_abs_dbetae=max_db)


def check_frozen_background_many(run_dict):
    """Apply check_frozen_background to multiple runs."""
    rows=[check_frozen_background(out,key=key) for key,out in run_dict.items()]
    return pd.DataFrame(rows) if pd is not None else rows


def _get_time_days_from_run(out):
    for key in ("t_days","time_days"):
        if key in out: return np.asarray(out[key],dtype=float)
    return np.arange(out["psi_time"].shape[1],dtype=float)


def project_psi_time_onto_eigenmode(out, eig_out, mode_rank=0, psi_stationary=None,
                                    t0_day=20, t1_day=200,
                                    subtract_window_mean_if_no_stationary=True):
    """
    Project complex psi_time onto one eigenmode to get a complex coefficient a(t).
    The phase of a(t) gives the true free-mode period (not T/2 from amplitude diagnostics).

    Returns
    -------
    proj : dict  t_days, mask, coeff, coeff_window, t_window, psi_res, mode, mode_rank,
                 mode_id, free_mode_period_days, omega, omega_r, omega_i.
    """
    psi_time=np.asarray(out["psi_time"],dtype=np.complex128)
    imax,n_time=psi_time.shape
    t_days=_get_time_days_from_run(out)
    mask=(t_days>=t0_day)&(t_days<=t1_day)
    if np.sum(mask)<5: raise ValueError("Too few time points in selected window.")
    info=get_eigenmode_info(eig_out,mode_rank); A=info["mode"]
    # stationary component
    if psi_stationary is None:
        psi_s=np.nanmean(psi_time[:,mask],axis=1) if subtract_window_mean_if_no_stationary else np.zeros(imax,dtype=np.complex128)
    else:
        psi_s=np.asarray(psi_stationary,dtype=np.complex128)
        if psi_s.ndim==2: psi_s=psi_s[:,-1]
    psi_res=psi_time-psi_s[:,None]
    denom=np.vdot(A,A)
    coeff=np.array([np.vdot(A,psi_res[:,it])/denom for it in range(n_time)])
    return dict(t_days=t_days,mask=mask,coeff=coeff,coeff_window=coeff[mask],t_window=t_days[mask],
                psi_res=psi_res,psi_stationary=psi_s,mode=A,mode_rank=mode_rank,mode_id=info["mode_id"],
                free_mode_period_days=info["period_days"],omega=info["omega"],
                omega_r=info["omega_r"],omega_i=info["omega_i"])


def period_from_complex_phase(t_days, coeff, min_amp_frac=0.05):
    """
    Estimate period from the unwrapped phase of a complex coefficient.
    Avoids the T/2 half-period artifact from amplitude-like diagnostics.

    Returns
    -------
    dict  period_days, slope_rad_per_day, intercept, n_points, phase, good.
    """
    t_days=np.asarray(t_days,dtype=float); coeff=np.asarray(coeff,dtype=np.complex128)
    amp=np.abs(coeff); amp_max=np.nanmax(amp)
    good_amp=amp>min_amp_frac*amp_max if (np.isfinite(amp_max) and amp_max>0) else np.ones(len(amp),dtype=bool)
    good=np.isfinite(t_days)&np.isfinite(coeff.real)&np.isfinite(coeff.imag)&good_amp
    nan_result=dict(period_days=np.nan,slope_rad_per_day=np.nan,intercept=np.nan,
                    n_points=int(np.sum(good)),phase=None,good=good)
    if np.sum(good)<5: return nan_result
    phase=np.unwrap(np.angle(coeff[good]))
    slope,intercept=np.polyfit(t_days[good],phase,1)
    T=2.0*np.pi/np.abs(slope) if slope!=0 else np.inf
    return dict(period_days=T,slope_rad_per_day=slope,intercept=intercept,
                n_points=int(np.sum(good)),phase=phase,good=good)


def fft_peak_period(x, dt_days=1.0, search_band=(10.0,80.0),
                    zero_pad_factor=8, remove_mean=True):
    """Estimate the dominant FFT period in a band. Can return T/2 for amplitude-like inputs."""
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)<5: return dict(period_days=np.nan,power=np.nan)
    if remove_mean: x=x-np.nanmean(x)
    win=np.hanning(len(x)); xw=x*win
    nfft=int(2**np.ceil(np.log2(max(len(xw)*zero_pad_factor,len(xw)))))
    freq=np.fft.rfftfreq(nfft,d=dt_days); spec=np.abs(np.fft.rfft(xw))
    freq=freq[1:]; spec=spec[1:]; period=1.0/freq
    mask=np.isfinite(period)&(period>=search_band[0])&(period<=search_band[1])
    if np.sum(mask)==0: return dict(period_days=np.nan,power=np.nan)
    pb=period[mask]; sb=spec[mask]; im=int(np.argmax(sb))
    return dict(period_days=float(pb[im]),power=float(sb[im]))


def diagnose_free_mode_period_from_run(out, eig_out, mode_rank=0,
                                       psi_stationary=None, t0_day=20, t1_day=200,
                                       search_band=(10.0,80.0), key=None, extra_series=None):
    """
    Full diagnostic for one HM76 run: phase-based period + FFT of amplitude-like quantities.

    Returns
    -------
    row : dict   Summary statistics.
    proj : dict  Projection output from project_psi_time_onto_eigenmode.
    """
    key=key or out.get("key","run")
    proj=project_psi_time_onto_eigenmode(out,eig_out,mode_rank=mode_rank,
                                          psi_stationary=psi_stationary,t0_day=t0_day,t1_day=t1_day)
    t_w=proj["t_window"]; a=proj["coeff_window"]; free_T=proj["free_mode_period_days"]
    ph=period_from_complex_phase(t_w,a)
    phase_T=ph["period_days"]
    fRe=fft_peak_period(np.real(a),dt_days=1.0,search_band=search_band)
    fIm=fft_peak_period(np.imag(a),dt_days=1.0,search_band=search_band)
    fAbsRe=fft_peak_period(np.abs(np.real(a)),dt_days=1.0,search_band=search_band)
    fReq=fft_peak_period(np.real(a)**2,dt_days=1.0,search_band=search_band)
    fAbs=fft_peak_period(np.abs(a),dt_days=1.0,search_band=search_band)
    row=dict(key=key,tau_days=out.get("tau_days",np.nan),feedback=out.get("feedback","unknown"),
             wave_mean_feedback=out.get("wave_mean_feedback",np.nan),
             mean_flow_coupling=out.get("mean_flow_coupling",np.nan),
             mode_rank=mode_rank,mode_id=proj["mode_id"],t0_day=t0_day,t1_day=t1_day,
             free_mode_period_days=free_T,phase_period_days=phase_T,
             phase_period_over_free=phase_T/free_T if np.isfinite(free_T) else np.nan,
             phase_slope_rad_per_day=ph["slope_rad_per_day"],n_phase_points=ph["n_points"],
             fft_Re_coeff_days=fRe["period_days"],fft_Im_coeff_days=fIm["period_days"],
             fft_absRe_coeff_days=fAbsRe["period_days"],fft_Re_coeff_sq_days=fReq["period_days"],
             fft_abs_coeff_days=fAbs["period_days"])
    if extra_series:
        for name,series in extra_series.items():
            series=np.asarray(series,dtype=float)
            if len(series)!=len(proj["t_days"]): raise ValueError(f"extra_series '{name}' length mismatch.")
            fe=fft_peak_period(series[proj["mask"]],dt_days=1.0,search_band=search_band)
            sn=name.replace(" ","_").replace("/","_").replace("-","_").replace("^","pow")
            row[f"fft_{sn}_days"]=fe["period_days"]
    return row, proj


def diagnose_free_mode_period_many(run_dict, eig_out, mode_rank=0,
                                   psi_stationary_dict=None, t0_day=20, t1_day=200,
                                   search_band=(10.0,80.0), extra_series_dict=None):
    """Apply diagnose_free_mode_period_from_run to multiple runs."""
    rows=[]; projections={}
    for key,out in run_dict.items():
        psi_s=psi_stationary_dict.get(key) if psi_stationary_dict else None
        ex=extra_series_dict.get(key) if extra_series_dict else None
        row,proj=diagnose_free_mode_period_from_run(out,eig_out,mode_rank=mode_rank,
                                                     psi_stationary=psi_s,t0_day=t0_day,
                                                     t1_day=t1_day,search_band=search_band,
                                                     key=key,extra_series=ex)
        rows.append(row); projections[key]=proj
    table=pd.DataFrame(rows) if pd is not None else rows
    return table, projections


def reconstruct_stationary_plus_free_mode(psi_stationary, eig_out, mode_rank=0,
                                          amp_free=0.3, phase0=0.0,
                                          n_days=250, level_km=30.0, plot=True):
    """
    Reconstruct idealized interference: psi(z,t) = psi_s(z) + amp_free*A(z)*exp(-i*omega*t).
    Returns phase period and amplitude-like FFT periods for comparison.
    """
    import matplotlib.pyplot as plt
    z=eig_out["z"]; info=get_eigenmode_info(eig_out,mode_rank)
    A=info["mode"]; omega_m=info["omega"]; T_free=info["period_days"]; c=info["c_phase"]
    psi_s=np.asarray(psi_stationary,dtype=np.complex128)
    if psi_s.ndim==2: psi_s=psi_s[:,-1]
    if len(psi_s)!=len(z): raise ValueError("psi_stationary length mismatch.")
    mx=np.nanmax(np.abs(psi_s))
    if mx>0: psi_s_n=psi_s/mx
    else: psi_s_n=psi_s.copy()
    t_d=np.arange(n_days,dtype=float); t_s=t_d*86400.0
    free_c=amp_free*np.exp(-1j*omega_m*t_s+1j*phase0)
    psi_rec=np.zeros((len(z),n_days),dtype=np.complex128)
    psi_free=np.zeros((len(z),n_days),dtype=np.complex128)
    for it in range(n_days):
        psi_free[:,it]=free_c[it]*A; psi_rec[:,it]=psi_s_n+psi_free[:,it]
    lev=np.argmin(np.abs(z/1000.0-level_km))
    f_abs=fft_peak_period(np.abs(psi_rec[lev,:]),dt_days=1.0,search_band=(5.0,100.0))
    f_sq=fft_peak_period(np.real(psi_free[lev,:])**2,dt_days=1.0,search_band=(5.0,100.0))
    return dict(t_days=t_d,psi_rec=psi_rec,psi_free=psi_free,free_coeff=free_c,
                mode_rank=mode_rank,mode_id=info["mode_id"],omega=omega_m,
                free_mode_period_days=T_free,c_phase=c,
                fft_abs_total_period_days=f_abs["period_days"],
                fft_free_real_sq_period_days=f_sq["period_days"])
