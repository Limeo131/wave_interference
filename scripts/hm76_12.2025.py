# %% [markdown]
# # Stationary Run

# %%
def run_hm76_stationary(hb, tau,
                        dz=1000.0,
                        imax=71,
                        dt=360.0*15.0,
                        n_days=300,
                        alpha_on=True,
                        verbose=False):
    """
    Linear HM76 run to obtain the stationary forced mode:
    - wave–mean flow interaction OFF (background wind fixed)
    - radiative damping ON (alpha_on=True)
    - bottom forcing slowly switched on with time scale tau.

    Parameters
    ----------
    hb : float
        Base-level forcing amplitude (m), appears in psi(0, t).
    tau : float
        Spin-up time scale for forcing (seconds), i.e. psi ~ (1 - exp(-t/tau)).
    dz, imax, dt, n_days, alpha_on, verbose : same as run_hm76

    Returns
    -------
    out : dict
        Includes:
        - 'z'   : altitude (m)
        - 'ud'  : background u(z), repeated in time (shape imax × n_days)
        - 'epz','tusu', diagnostics as in run_hm76
        - 'psi_stationary' : complex stationary mode ψ(z) at the end of run
        - 'u_bg' : background wind profile u_bg(z)
    """

    # -----------------
    # model parameters
    # -----------------
    s = 2.0
    mmax = n_days * int(3600*24.0/dt)
    idd  = int((mmax - 1) / int(3600*24.0/dt))

    a = 6378000.0
    k = s/(a*np.cos(np.pi/3.0))
    l = 3.0/a
    omega = 7.29e-5
    g = 9.81
    f0 = 2.0*omega*np.sin(np.pi/3.0)
    beta = 2.0*omega*np.cos(np.pi/3.0)/a
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
        print('STATIONARY run: tau =', int(tau/3600/24), 'days, hb =', hb)

    # -----------------------------
    # allocate arrays
    # -----------------------------
    z     = np.zeros(imax)
    alpha = np.zeros(imax)
    u0    = np.zeros((imax, 4))                 # keep this = background
    ur    = np.zeros(imax)
    qy    = np.zeros((imax, 4))                 # same: kept = background PV gradient
    psi   = np.zeros((imax, 4), dtype=np.complex64)
    q     = np.zeros((imax, 4), dtype=np.complex64)

    ak = np.zeros(imax)
    bk = np.zeros(imax, dtype=np.complex64)

    ud   = np.zeros((imax, idd+1))
    epz  = np.zeros((imax, idd+1))
    tusu = np.zeros((imax, idd+1))
    psi_time = np.zeros((imax, idd+1), dtype=np.complex64)
    q_time   = np.zeros((imax, idd+1), dtype=np.complex64)

    def diag_array():
        arr = np.zeros(idd+1)
        arr[:] = np.nan
        return arr

    ubar25 = diag_array(); betae25 = diag_array(); wa25 = diag_array()
    ua25   = diag_array(); epz25   = diag_array()
    ubar30 = diag_array(); betae30 = diag_array(); wa30 = diag_array()
    ua30   = diag_array(); epz30   = diag_array()
    ubar35 = diag_array(); betae35 = diag_array(); wa35 = diag_array()
    ua35   = diag_array(); epz35   = diag_array()
    ubar40 = diag_array(); betae40 = diag_array(); wa40 = diag_array()
    ua40   = diag_array(); epz40   = diag_array()

    # -----------------------------
    # set z, alpha, ur
    # -----------------------------
    for i in range(imax):
        z[i] = dz*i + 10000.0
        if alpha_on:
            alpha[i] = (1.5 + np.tanh((z[i]-35000.0)/h0))*1.0e-6
        else:
            alpha[i] = 0.0
        ur[i] = 12.0 + 0.003*(z[i]-10000.0)

    # -----------------------------
    # background wind u_bg and qy_bg
    # -----------------------------
    u_bg = np.zeros(imax)
    for i in range(imax):
        # same initial profile as full model
        u_bg[i] = 12.0 + (z[i]-10000.0)*(52.0-12.0)/40000.0
        u0[i, :] = u_bg[i]

    qy_bg = np.zeros(imax)
    for i in range(1, imax-1):
        qy_bg[i] = -l*l*u_bg[i] + (f0*f0/ensq)*(
            (u_bg[i+1] + u_bg[i-1] - 2.0*u_bg[i])/(dz*dz)
            - (u_bg[i+1] - u_bg[i-1])/(2.0*dz*h0)
        )
        qy[i, :] = qy_bg[i]
        psi[i, :] = 0.0 + 0.0j
        q[i,   :] = 0.0 + 0.0j

    # -----------------------------
    # Time stepping (AB3 for wave only)
    # -----------------------------
    m  = 0
    mm = 0
    i25 = i30 = i35 = i40 = 0
    steps_per_day = int(3600*24.0/dt)

    while m < mmax:
        t = dt * m

        # Fix background wind and qy (wave–mean interaction OFF)
        u0[:, 0] = u_bg
        u0[:, 1] = u_bg
        u0[:, 2] = u_bg
        qy[:, 0] = qy_bg
        qy[:, 1] = qy_bg
        qy[:, 2] = qy_bg

        # ---- q, qy tend ----
        i = 1
        while i < imax-1:
            # effective beta from fixed qy_bg
            betae1 = beta - eps*qy[i,0]
            betae2 = beta - eps*qy[i,1]
            betae3 = beta - eps*qy[i,2]

            # advection by fixed u_bg
            uadv1 = -(0.0+1.0j)*k*eps*u0[i,0]*q[i,0]
            uadv2 = -(0.0+1.0j)*k*eps*u0[i,1]*q[i,1]
            uadv3 = -(0.0+1.0j)*k*eps*u0[i,2]*q[i,2]

            vadv1 = -betae1*(0.0+1.0j)*k*psi[i,0]
            vadv2 = -betae2*(0.0+1.0j)*k*psi[i,1]
            vadv3 = -betae3*(0.0+1.0j)*k*psi[i,2]

            # radiative damping on the wave (kept!)
            damp0 = (f0*f0/ensq)*(0.5/h0)*alpha[i]*(
                (psi[i+1,2]-psi[i-1,2])/(2.0*dz) + psi[i,2]/(2.0*h0)
            )
            damp1 = -(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2 = +(f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3 = -(f0*f0/(ensq*dz*h0))*0.125
            damp3 = damp3*((alpha[i+1]+alpha[i])*psi[i+1,2]
                           -(alpha[i-1]+alpha[i])*psi[i-1,2]
                           +(alpha[i+1]-alpha[i-1])*psi[i,2])

            q[i,3] = q[i,2] + (dt/12.0)*(
                5.0*uadv1 - 16.0*uadv2 + 23.0*uadv3
                + 5.0*vadv1 - 16.0*vadv2 + 23.0*vadv3
            )
            q[i,3] = q[i,3] + dt*(damp0 + damp1 + damp2 + damp3)

            i += 1

        # ---- invert psi from q (same as full model) ----
        ak[imax-2] = 0.0
        bk[imax-2] = 0.0 + 0.0j
        i = imax-2
        while i > -1:
            ak[i-1] = -1.0/(ak[i] - p)
            bk[i-1] = (r*q[i,3] - bk[i])/(ak[i] - p)
            i -= 1

        psi[0,3] = hb*(g/f0)*(1.0 - np.exp(-t/tau))   # lower BC

        i = 0
        while i < imax-1:
            psi[i+1,3] = ak[i]*psi[i,3] + bk[i]
            i += 1

        # ---- shift time levels for q, psi ----
        q[:,0]   = q[:,1];   q[:,1]   = q[:,2];   q[:,2]   = q[:,3]
        psi[:,0] = psi[:,1]; psi[:,1] = psi[:,2]; psi[:,2] = psi[:,3]

        # ---- daily output ----
        if m % steps_per_day == 0:
            day = t/(3600.0*24.0)
            if verbose:
                print("stationary run m =", m, " time =", day, "days")

            # mean wind is fixed background
            ud[:,mm] = u_bg[:]
            psi_time[:, mm] = psi[:, 2]
            q_time[:, mm]   = q[:, 2]

            i = 1
            while i < imax-1:
                epz[i,mm] = -0.5*k*eps*np.imag(
                    psi[i,1]*np.conj(psi[i+1,1]-psi[i-1,1])/(2.0*dz)
                )
                tusu[i,mm] = eps*ud[i,mm]/(beta-eps*qy[i,1]) - 1.0/(k**2 + l**2 + lrmsq)
                i += 1

            # 25 km
            if i25 == 0:
                ubar25[mm]  = u_bg[16]
                betae25[mm] = beta - eps*qy_bg[16]
                aa25        = abs(q[16,1])
                wa25[mm]    = eps*aa25*aa25*np.exp(25.0/7.0)*0.25/betae25[mm]
                epz25[mm]   = -0.5*k*eps*np.imag(
                    psi[16,1]*np.conj(psi[17,1]-psi[15,1])/(2.0*dz)
                )
                ua25[mm]    = u_bg[16]
                if betae25[mm] < 0:
                    i25 = mm

            # 30 km
            if i30 == 0:
                ubar30[mm]  = u_bg[21]
                betae30[mm] = beta - eps*qy_bg[21]
                aa30        = abs(q[21,1])
                wa30[mm]    = eps*aa30*aa30*np.exp(30.0/7.0)*0.25/betae30[mm]
                epz30[mm]   = -0.5*k*eps*np.imag(
                    psi[21,1]*np.conj(psi[22,1]-psi[20,1])/(2.0*dz)
                )
                ua30[mm]    = u_bg[21]
                if betae30[mm] < 0:
                    i30 = mm

            # 35 km
            if i35 == 0:
                ubar35[mm]  = u_bg[26]
                betae35[mm] = beta - eps*qy_bg[26]
                aa35        = abs(q[26,1])
                wa35[mm]    = eps*aa35*aa35*np.exp(35.0/7.0)*0.25/betae35[mm]
                ua35[mm]    = u_bg[26]
                epz35[mm]   = -0.5*k*eps*np.imag(
                    psi[26,1]*np.conj(psi[27,1]-psi[25,1])/(2.0*dz)
                )
                if betae35[mm] < 0:
                    i35 = mm

            # 40 km
            if i40 == 0:
                ubar40[mm]  = u_bg[31]
                betae40[mm] = beta - eps*qy_bg[31]
                aa40        = abs(q[31,1])
                wa40[mm]    = eps*aa40*aa40*np.exp(40.0/7.0)*0.25/betae40[mm]
                epz40[mm]   = -0.5*k*eps*np.imag(
                    psi[31,1]*np.conj(psi[32,1]-psi[30,1])/(2.0*dz)
                )
                ua40[mm]    = u_bg[31]
                if betae40[mm] < 0:
                    i40 = mm

            mm += 1

        m += 1

    # -----------------------------
    # Scale EPZ (same as original)
    # -----------------------------
    c = f0*f0/(ensq*4.0e-3)
    for i in range(1, imax-1):
        # here ud is just u_bg, but keep scaling form
        epz[i, :] = 0.25*c*np.exp(z[i]/7000.0)*epz[i, :]/(ud[i,20]*ud[i,20])

    # final stationary mode: psi at last time level
    psi_stationary = psi[:,2].copy()

    out = dict(
        z=z, ud=ud, epz=epz, tusu=tusu,
        ubar25=ubar25, betae25=betae25, wa25=wa25, ua25=ua25, epz25=epz25,
        ubar30=ubar30, betae30=betae30, wa30=wa30, ua30=ua30, epz30=epz30,
        ubar35=ubar35, betae35=betae35, wa35=wa35, ua35=ua35, epz35=epz35,
        ubar40=ubar40, betae40=betae40, wa40=wa40, ua40=ua40, epz40=epz40,
        hb=hb, tau=tau, dt=dt, dz=dz,
        psi_stationary=psi_stationary,psi_time=psi_time,q_time=q_time,
        u_bg=u_bg
    )
    return out

# %% [markdown]
# # Transient Run

# %%
import numpy as np

def run_hm76_transient(dz=1000.0,
                        imax=71,
                        dt=360.0*15.0,
                        n_days=300,
                        alpha_on=True,
                        verbose=False,
                        z0=30000.0,
                        sigma=10000.0,
                        amp=1.0):
    """
    Linear HM76 run to obtain the least-damped transient (traveling) mode:
    - wave–mean flow interaction OFF (background wind fixed)
    - radiative damping ON/OFF via alpha_on
    - NO bottom forcing (free wave)
    - start from an arbitrary localized initial perturbation in q

    Parameters
    ----------
    dz, imax, dt, n_days : grid and time settings
    alpha_on : bool
        If True, include radiative damping in the wave equation.
    verbose : bool
        If True, print progress.
    z0 : float
        Center height (m) of initial perturbation (default 30 km).
    sigma : float
        Width (m) of initial Gaussian perturbation.
    amp : float
        Amplitude of initial perturbation.

    Returns
    -------
    out : dict
        - 'z'        : altitude (m)
        - 'ud'       : background u(z) (repeated in time), shape (imax, n_days)
        - 'psi_last' : complex ψ(z) at final time (approx. transient mode)
        - 'psi_time' : ψ(z, t) at each daily output, shape (imax, n_days)
        - 'u_bg'     : background wind profile u_bg(z)
        - 'qy_bg'    : background PV gradient (-q_y sans beta)
        - 'k','l','f0','beta','dt','dz'
    """

    # -------------
    # parameters
    # -------------
    s = 2.0
    mmax = n_days * int(3600*24.0/dt)
    idd  = int((mmax - 1) / int(3600*24.0/dt))

    a = 6378000.0
    k = s/(a*np.cos(np.pi/3.0))
    l = 3.0/a
    omega = 7.29e-5
    g = 9.81
    f0 = 2.0*omega*np.sin(np.pi/3.0)
    beta = 2.0*omega*np.cos(np.pi/3.0)/a
    ensq = 4.0e-4
    h0 = 7000.0
    urz = 3.0e-3  # not used here, kept for completeness
    eps = 8.0/(3.0*np.pi)
    lrmsq = (f0*f0)/(4.0*ensq*h0*h0)
    r = (ensq*dz*dz)/(f0*f0)
    p = 2.0 + r*(k*k + l*l + lrmsq)

    if verbose:
        print('TRANSIENT run: n_days =', n_days, 'dt (day) =', dt/86400.0)

    # -------------
    # allocate
    # -------------
    imax = int(imax)
    z     = np.zeros(imax)
    alpha = np.zeros(imax)
    u0    = np.zeros((imax, 4))
    ur    = np.zeros(imax)
    qy    = np.zeros((imax, 4))
    psi   = np.zeros((imax, 4), dtype=np.complex64)
    q     = np.zeros((imax, 4), dtype=np.complex64)

    ak = np.zeros(imax)
    bk = np.zeros(imax, dtype=np.complex64)

    ud        = np.zeros((imax, idd+1))
    psi_time  = np.zeros((imax, idd+1), dtype=np.complex64)
    q_time    = np.zeros((imax, idd+1), dtype=np.complex64)

    # -------------
    # z, alpha, ur
    # -------------
    for i in range(imax):
        z[i] = dz*i + 10000.0
        if alpha_on:
            alpha[i] = (1.5 + np.tanh((z[i]-35000.0)/h0))*1.0e-6
        else:
            alpha[i] = 0.0
        ur[i] = 12.0 + 0.003*(z[i]-10000.0)

    # -------------
    # background u and qy
    # -------------
    u_bg = np.zeros(imax)
    for i in range(imax):
        u_bg[i] = 12.0 + (z[i]-10000.0)*(52.0-12.0)/40000.0
        u0[i, :] = u_bg[i]

    qy_bg = np.zeros(imax)
    for i in range(1, imax-1):
        qy_bg[i] = -l*l*u_bg[i] + (f0*f0/ensq)*(
            (u_bg[i+1] + u_bg[i-1] - 2.0*u_bg[i])/(dz*dz)
            - (u_bg[i+1] - u_bg[i-1])/(2.0*dz*h0)
        )
        qy[i, :] = qy_bg[i]

    # -------------
    # initial perturbation in q (AB3 startup: time levels 0,1,2 identical)
    # -------------
    for i in range(1, imax-1):
        zz = z[i]
        q_init = amp * np.exp(-0.5*((zz - z0)/sigma)**2)
        q[i,0] = q_init + 0.0j
        q[i,1] = q_init + 0.0j
        q[i,2] = q_init + 0.0j

    # Use q(t=2) to invert psi once as the initial structure
    ak[imax-2] = 0.0
    bk[imax-2] = 0.0 + 0.0j
    i = imax-2
    while i > -1:
        ak[i-1] = -1.0/(ak[i] - p)
        bk[i-1] = (r*q[i,2] - bk[i])/(ak[i] - p)
        i -= 1
    psi[0,2] = 0.0 + 0.0j   # free wave: lower boundary ψ(0)=0
    i = 0
    while i < imax-1:
        psi[i+1,2] = ak[i]*psi[i,2] + bk[i]
        i += 1

    # At the start, set psi identical at the three time levels
    psi[:,0] = psi[:,2]
    psi[:,1] = psi[:,2]

    # -------------
    # time stepping
    # -------------
    m  = 0
    mm = 0
    steps_per_day = int(3600*24.0/dt)

    while m < mmax:
        t = dt * m

        # Fix background wind / qy (wave–mean interaction OFF)
        u0[:,0] = u_bg
        u0[:,1] = u_bg
        u0[:,2] = u_bg
        qy[:,0] = qy_bg
        qy[:,1] = qy_bg
        qy[:,2] = qy_bg

        # ---- AB3 stepping for q, psi (wave equation + radiative damping only)----
        i = 1
        while i < imax-1:
            betae1 = beta - eps*qy[i,0]
            betae2 = beta - eps*qy[i,1]
            betae3 = beta - eps*qy[i,2]

            uadv1 = -(0.0+1.0j)*k*eps*u0[i,0]*q[i,0]
            uadv2 = -(0.0+1.0j)*k*eps*u0[i,1]*q[i,1]
            uadv3 = -(0.0+1.0j)*k*eps*u0[i,2]*q[i,2]

            vadv1 = -betae1*(0.0+1.0j)*k*psi[i,0]
            vadv2 = -betae2*(0.0+1.0j)*k*psi[i,1]
            vadv3 = -betae3*(0.0+1.0j)*k*psi[i,2]

            damp0 = (f0*f0/ensq)*(0.5/h0)*alpha[i]*(
                (psi[i+1,2]-psi[i-1,2])/(2.0*dz) + psi[i,2]/(2.0*h0)
            )
            damp1 = -(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2 = +(f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3 = -(f0*f0/(ensq*dz*h0))*0.125
            damp3 = damp3*((alpha[i+1]+alpha[i])*psi[i+1,2]
                           -(alpha[i-1]+alpha[i])*psi[i-1,2]
                           +(alpha[i+1]-alpha[i-1])*psi[i,2])

            q[i,3] = q[i,2] + (dt/12.0)*(
                5.0*uadv1 - 16.0*uadv2 + 23.0*uadv3
                + 5.0*vadv1 - 16.0*vadv2 + 23.0*vadv3
            )
            q[i,3] = q[i,3] + dt*(damp0 + damp1 + damp2 + damp3)
            i += 1

        # ---- invert psi from q (free wave: lower boundary ψ(0)=0) ----
        ak[imax-2] = 0.0
        bk[imax-2] = 0.0 + 0.0j
        i = imax-2
        while i > -1:
            ak[i-1] = -1.0/(ak[i] - p)
            bk[i-1] = (r*q[i,3] - bk[i])/(ak[i] - p)
            i -= 1

        psi[0,3] = 0.0 + 0.0j
        i = 0
        while i < imax-1:
            psi[i+1,3] = ak[i]*psi[i,3] + bk[i]
            i += 1

        # ---- shift time levels ----
        q[:,0]   = q[:,1];   q[:,1]   = q[:,2];   q[:,2]   = q[:,3]
        psi[:,0] = psi[:,1]; psi[:,1] = psi[:,2]; psi[:,2] = psi[:,3]

        # ---- store ψ and background wind once per day ----
        if m % steps_per_day == 0:
            day = t/86400.0
            if verbose:
                print("transient run m =", m, " time =", day, "days")
            ud[:,mm]       = u_bg[:]
            psi_time[:,mm] = psi[:,2]
            q_time[:,mm]   = q[:,2]
            mm += 1

        m += 1

    psi_last = psi[:,2].copy()

    out = dict(
        z=z,
        ud=ud,
        psi_last=psi_last,
        psi_time=psi_time,
        q_time=q_time,
        u_bg=u_bg,
        qy_bg=qy_bg,
        k=k,
        l=l,
        f0=f0,
        beta=beta,
        dt=dt,
        dz=dz
    )
    return out


# %%
import numpy as np

def extract_transient_mode(psi_time, z, k,
                           dt_out_days=1.0,
                           n_fit=None,
                           level_ref=None,
                           verbose=False,
                           q_time=None):
    """
    Extract the least-damped transient mode from psi_time(z, t),
    and optionally extract the corresponding q_mode from q_time(z, t).

    Parameters
    ----------
    psi_time : ndarray, shape (nz, nt)
        Complex streamfunction ψ(z, t), e.g. run_hm76_transient['psi_time'].
    z : ndarray, shape (nz,)
        Height array (m).
    k : float
        Zonal wavenumber (1/m).
    dt_out_days : float, optional
        Temporal resolution of psi_time in days (default: 1 day per snapshot).
    n_fit : int, optional
        Number of last timesteps used for phase fitting and averaging.
        If None, use the last 1/3 of timesteps (at least 4).
    level_ref : int, optional
        Reference height index used to track the phase.
        If None, automatically choose the level with the largest time-mean |psi|
        in the fitting window.
    verbose : bool, optional
        If True, print diagnostics.
    q_time : ndarray, shape (nz, nt), optional
        If provided, q_time will be phase-aligned in the same way as psi_time
        and averaged to give q_mode; otherwise q_mode=None.

    Returns
    -------
    out : dict
        - 'psi_mode' : vertical structure (normalized so that max|ψ|=1), shape (nz,)
        - 'q_mode'   : if q_time is not None, the corresponding vertical structure
                       of q, using the same normalization factor as psi_mode,
                       shape (nz,); otherwise None.
        - 'c_phase'  : phase speed (m/s)
        - 'omega'    : angular frequency (rad/s)
        - 'level_ref': reference level index
        - 't_fit'    : time axis (s) used for fitting, shape (n_fit,)
        - 'phase_fit': unwrapped phase (rad) used for fitting, shape (n_fit,)
    """
    psi_time = np.asarray(psi_time)
    z = np.asarray(z)

    nz, nt = psi_time.shape
    if n_fit is None:
        n_fit = max(nt // 3, 4)   # at least 4 points

    if n_fit > nt:
        raise ValueError("n_fit cannot be larger than nt")

    # Fitting window: last n_fit timesteps
    i_start = nt - n_fit
    i_end   = nt
    psi_win = psi_time[:, i_start:i_end]   # (nz, n_fit)

    if q_time is not None:
        q_time = np.asarray(q_time)
        if q_time.shape != psi_time.shape:
            raise ValueError("q_time shape must match psi_time shape")
        q_win = q_time[:, i_start:i_end]   # (nz, n_fit)
    else:
        q_win = None

    # ---- Automatically select reference level: max time-mean |psi| in window ----
    if level_ref is None:
        amp_mean = np.mean(np.abs(psi_win), axis=1)
        level_ref = np.argmax(amp_mean)
        if verbose:
            print("Auto-selected level_ref index =", level_ref,
                  " z ≈", z[level_ref]/1000.0, "km")

    # ---- 1) Estimate phase speed from phase at reference level vs time ----
    dt_out_sec = dt_out_days * 86400.0
    t_fit = np.arange(i_start, i_end) * dt_out_sec   # (n_fit,)

    psi_ref = psi_time[level_ref, i_start:i_end]     # (n_fit,)
    phase = np.angle(psi_ref)
    phase_unwrap = np.unwrap(phase)

    # Linear fit phase(t) = a*t + b, slope a ≈ dφ/dt
    a, b = np.polyfit(t_fit, phase_unwrap, 1)
    omega = -a                      # exp(i(kx - ωt)) → dφ/dt = -ω
    c_phase = omega / k             # m/s

    if verbose:
        print("Fitted omega (rad/s) =", omega)
        print("Phase speed c_phase (m/s) =", c_phase)

    # ---- 2) Phase-aligned time average to obtain modal structure ----
    psi_aligned = np.zeros_like(psi_win, dtype=np.complex128)
    if q_win is not None:
        q_aligned = np.zeros_like(q_win, dtype=np.complex128)
    else:
        q_aligned = None

    for j in range(n_fit):
        # Use the reference level as phase reference and rotate to zero phase
        phase_ref_j = np.angle(psi_win[level_ref, j])
        rot = np.exp(-1j * phase_ref_j)

        psi_aligned[:, j] = psi_win[:, j] * rot
        if q_aligned is not None:
            q_aligned[:, j] = q_win[:, j] * rot

    # Time average
    psi_mode_raw = np.mean(psi_aligned, axis=1)   # (nz,)
    if q_aligned is not None:
        q_mode_raw = np.mean(q_aligned, axis=1)   # (nz,)
    else:
        q_mode_raw = None

    # ---- 3) Normalize ψ and q with the same factor ----
    max_amp = np.max(np.abs(psi_mode_raw))
    if max_amp > 0:
        psi_mode = psi_mode_raw / max_amp
        if q_mode_raw is not None:
            q_mode = q_mode_raw / max_amp
        else:
            q_mode = None
    else:
        psi_mode = psi_mode_raw
        q_mode = q_mode_raw

    out = dict(
        psi_mode=psi_mode,
        q_mode=q_mode,
        c_phase=c_phase,
        omega=omega,
        level_ref=level_ref,
        t_fit=t_fit,
        phase_fit=phase_unwrap
    )
    return out


# %% [markdown]
# # Run stationary & transient

# %%
# ==== 1) run stationary mode ====
hb  = 3500.0              # bottom forcing amplitude you want
tau = 40 * 24 * 3600.0    # relatively slow ramp-up, e.g., 40 days

out_stat = run_hm76_stationary(
    hb=hb,
    tau=tau,
    dz=1000.0,
    imax=71,
    dt=360.0*15.0,
    n_days=800,          # integrate long enough so the mode fully converges
    alpha_on=True,       # radiative damping ON
    verbose=False
)

z        = out_stat['z']                 # height (m)
psi_stat = out_stat['psi_stationary']    # stationary mode ψ(z) (complex)
psi_s = psi_stationary                      # shape (nz,)
q_time_stat = out_stat['q_time']           # shape (nz, nt_stat)
q_s = q_time_stat[:, -1]                   # q(z) on the last day, shape (nz,)

# ==== 2) run transient mode ====
out_tr = run_hm76_transient(
    dz=1000.0,
    imax=71,
    dt=360.0*15.0,
    n_days=800,          # also integrate long enough here
    alpha_on=True,
    verbose=False,
    z0=30000.0,          # center height of initial perturbation (~30 km)
    sigma=10000.0,
    amp=1.0
)

psi_time_tr = out_tr['psi_time']   # (nz, nt_tr)
q_time_tr   = out_tr['q_time']     # (nz, nt_tr)
k           = out_tr['k']

mode_info = extract_transient_mode(
    psi_time=psi_time_tr,
    q_time=q_time_tr,     
    z=z,
    k=k,
    dt_out_days=1.0,
    n_fit=None,
    level_ref=None,
    verbose=True
)

psi_tr = mode_info['psi_mode']     # (nz,)
q_tr   = mode_info['q_mode']       # (nz,) — aligned with psi_tr, same normalization
c_tr   = mode_info['c_phase']
omega  = mode_info['omega']
iref   = mode_info['level_ref']

print("Transient phase speed c ≈ %.2f m/s" % c_tr)


# %% [markdown]
# # Plot

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_stationary_and_transient_modes(z, psi_stat, psi_tr,
                                        title_suffix=''):
    """
    z        : height (m), 1D array
    psi_stat : stationary mode ψ(z) (complex)
    psi_tr   : transient mode ψ(z) (complex, usually already normalized)
    """

    # ---- normalize amplitudes (for easier comparison) ----
    amp_stat = np.abs(psi_stat)
    amp_tr   = np.abs(psi_tr)

    # divide by each mode's own maximum
    if amp_stat.max() > 0:
        amp_stat = amp_stat / amp_stat.max()
    if amp_tr.max() > 0:
        amp_tr = amp_tr / amp_tr.max()

    phase_stat = np.angle(psi_stat)
    phase_tr   = np.angle(psi_tr)

    z_km = z / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(7,5), sharey=True)

    # --- left panel: amplitude vs z ---
    ax = axes[0]
    ax.plot(amp_stat, z_km, label='Stationary mode')
    ax.plot(amp_tr,   z_km, label='Transient mode', linestyle='--')
    ax.set_xlabel('Normalized |psi|')
    ax.set_ylabel('Height (km)')
    ax.set_title('Mode amplitude' + title_suffix)
    ax.grid(True, linestyle=':')
    ax.legend()

    # --- right panel: phase vs z ---
    ax = axes[1]
    ax.plot(phase_stat, z_km, label='Stationary mode')
    ax.plot(phase_tr,   z_km, label='Transient mode', linestyle='--')
    ax.set_xlabel('Phase (rad)')
    ax.set_title('Mode phase' + title_suffix)
    ax.grid(True, linestyle=':')
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_stationary_and_transient_modes(
    z=z,
    psi_stat=psi_stat,
    psi_tr=psi_tr,
    title_suffix=' (k=2)'
)


# %% [markdown]
# # Stationary sanity check

# %%
import numpy as np
import matplotlib.pyplot as plt

def sanity_check_stationary(out_stat,
                             levels_km=(20, 30, 40),
                             n_last_days=100):
    """
    out_stat    : output from run_hm76_stationary (must contain psi_time)
    levels_km   : heights (in km) to inspect
    n_last_days : number of last days to plot (to avoid early spin-up period)
    """
    z         = out_stat['z']
    psi_time  = out_stat['psi_time']   # shape (imax, nt)
    imax, nt  = psi_time.shape

    z_km = z / 1000.0
    t_days = np.arange(nt)   # each index corresponds to one day

    # Only look at the last n_last_days
    i_start = max(0, nt - n_last_days)
    t_win   = t_days[i_start:]
    psi_win = psi_time[:, i_start:]

    # Find indices corresponding to the requested heights
    level_idx = []
    for zk in levels_km:
        idx = np.argmin(np.abs(z_km - zk))
        level_idx.append(idx)

    plt.figure(figsize=(7, 6))

    # --- Top panel: amplitude ---
    ax1 = plt.subplot(2, 1, 1)
    for zk, idx in zip(levels_km, level_idx):
        amp = np.abs(psi_win[idx, :])
        ax1.plot(t_win, amp, label=f'{zk:.0f} km')
    ax1.set_ylabel('|psi|')
    ax1.set_title('Stationary run: amplitude vs time (last %d days)' % n_last_days)
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # --- Bottom panel: real part ---
    ax2 = plt.subplot(2, 1, 2)
    for zk, idx in zip(levels_km, level_idx):
        re_psi = np.real(psi_win[idx, :])
        ax2.plot(t_win, re_psi, label=f'{zk:.0f} km')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Re(psi)')
    ax2.set_title('Stationary run: real part vs time')
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    plt.show()


sanity_check_stationary(out_stat,
                        levels_km=np.array([20,30,40]),
                        n_last_days=800)


# %% [markdown]
# # Transient sanity check

# %%
def sanity_check_transient(out_tr,
                           mode_info,
                           n_last_days=150):
    """
    out_tr     : output from run_hm76_transient (must contain psi_time, z)
    mode_info  : output from extract_transient_mode (must contain level_ref)
    n_last_days: number of last days to plot
    """
    z         = out_tr['z']
    psi_time  = out_tr['psi_time']   # (imax, nt)
    imax, nt  = psi_time.shape

    z_km = z / 1000.0
    t_days = np.arange(nt)   # one output per day

    # reference level index
    iref = mode_info['level_ref']
    zref = z_km[iref]

    # only look at the last n_last_days
    i_start = max(0, nt - n_last_days)
    t_win   = t_days[i_start:]
    psi_ref = psi_time[iref, i_start:]

    amp  = np.abs(psi_ref)
    phase = np.angle(psi_ref)
    phase_unwrap = np.unwrap(phase)

    # Linear fit for phase(t) and ln(amp)(t)
    t_sec = t_win * 86400.0
    # phase fit: phase ≈ a*t + b  → slope a ~ dφ/dt
    a_phase, b_phase = np.polyfit(t_sec, phase_unwrap, 1)
    omega_est = -a_phase
    # amplitude decay: ln|psi| ≈ -γ t + const
    ln_amp = np.log(amp + 1e-30)
    a_ln, b_ln = np.polyfit(t_sec, ln_amp, 1)
    gamma_est = -a_ln   # decay rate

    # plotting
    plt.figure(figsize=(6,5))

    # --- top: phase vs time ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_win, phase_unwrap, label='phase (unwrapped)')
    # fitted line (converted back to days on x-axis)
    phase_fit = a_phase * t_sec + b_phase
    ax1.plot(t_win, phase_fit, '--', label='linear fit')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_title(f'Transient run @ z ≈ {zref:.1f} km: phase vs time (last {n_last_days} days)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # --- middle: |psi| vs time ---
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t_win, amp)
    ax2.set_ylabel('|psi|')
    ax2.set_title('Amplitude vs time')
    ax2.grid(True, linestyle=':')

    # --- bottom: ln|psi| vs time + linear fit ---
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_win, ln_amp, label='ln|psi|')
    ln_fit = a_ln * t_sec + b_ln
    ax3.plot(t_win, ln_fit, '--', label='linear fit')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('ln |psi|')
    ax3.set_title('ln amplitude vs time (check exponential decay)')
    ax3.grid(True, linestyle=':')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    print(f"Reference level: index = {iref}, z ≈ {zref:.1f} km")
    print(f"Estimated omega ≈ {omega_est:.3e} rad/s")
    print(f"Estimated decay rate gamma ≈ {gamma_est:.3e} 1/s")

sanity_check_transient(out_tr, mode_info, n_last_days=150)


# %%


# %%


# %% [markdown]
# # Standard HM76 Run

# %%
#This script attempts to reproduce Holton and Mass (1976)
from netCDF4 import Dataset 
import numpy as np 
import math
import scipy.stats as stats 
import matplotlib.pyplot as plot


def run_hm76(hb, tau,
             dz=1000.0,
             imax=71,
             dt=360.0*15.0,
             n_days=150,
             alpha_on=False,
             verbose=False):
    """
    Run the Holton & Mass (1976) style 1D model for given bottom forcing.

    Parameters
    ----------
    hb : float
        Base-level forcing amplitude (m), appears in psi(0, t).
    tau : float
        Spin-up time scale for forcing (seconds), i.e. exp(-t/tau).
    dz : float, optional
        Vertical grid spacing (m). Default 1000.
    imax : int, optional
        Number of vertical grid points. Default 71.
    dt : float, optional
        Time step (seconds). Default 360*30.
    n_days : int, optional
        Total integration length in days. Default 150.
    alpha_on : bool, optional
        If True, use the original HM76 alpha(z) vertical profile; if False, alpha=0.
    verbose : bool, optional
        If True, print progress.

    Returns
    -------
    out : dict
        Dictionary containing model outputs, including:
        - 'z'      : altitude array (m), shape (imax,)
        - 'ud'     : zonal mean wind (u0) saved once per day, shape (imax, n_days)
        - 'epz'    : EP flux proxy (epz), shape (imax, n_days)
        - 'tusu'   : turning surface array, shape (imax, n_days)
        - 'ubar25','ubar30','ubar35','ubar40' : time series at 25,30,35,40 km
        - 'betae25','betae30','betae35','betae40'
        - 'wa25','wa30','wa35','wa40'
        - 'ua25','ua30','ua35','ua40'
        - 'epz25','epz30','epz35','epz40'
    """

    # -----------------
    # model parameters
    # -----------------

    s = 2.0              # integer zonal wavenumber
    mmax = n_days * int(3600*24.0/dt)      # total time steps
    idd  = int((mmax - 1) / int(3600*24.0/dt))  # number of daily outputs (n_days-1，但+1后= n_days)

    a = 6378000.0        # Earth's radius
    k = s/(a*np.cos(np.pi/3.0))   # dimensional zonal wavenumber (1/m)
    l = 3.0/a                      # dimensional meridional wavenumber (1/m)
    omega = 7.29e-5      # Earth's rotation rate (1/s)
    g = 9.81             # gravitational acceleration (m/s^2)
    f0 = 2.0*omega*np.sin(np.pi/3.0)       # f at 60N  (1/s)
    beta = 2.0*omega*np.cos(np.pi/3.0)/a   # beta at 60N  (1/(m·s))
    ensq = 4.0e-4        # N^2 (1/s^2)
    h0 = 7000.0          # scale height (m)
    urz = 3.0e-3         # vertical shear for Ur (1/s)
    eps = 8.0/(3.0*np.pi)  # projection of sin^2 ly on sin ly
    lrmsq = (f0*f0)/(4.0*ensq*h0*h0)  # 1/R_R^2 (1/m^2)
    r = (ensq*dz*dz)/(f0*f0)
    p = 2.0 + r*(k*k + l*l + lrmsq)
    pu = 2.0 + r*l*l
    vp = 1.0 + 0.5*dz/h0
    vm = 1.0 - 0.5*dz/h0

    # if verbose:
    #     print("k, l, f0, beta = ", k, l, f0, beta)

    tau = round(tau,2)
    hb = round(hb,2)

    if verbose:
        print('tau=',int(tau/3600/24),'hb=',hb)


    # -----------------------------
    # allocate arrays
    # -----------------------------

    z     = np.zeros(imax)          # altitude (m)
    alpha = np.zeros(imax)          # radiative damping rate (1/s)
    u0    = np.zeros((imax, 4))     # zonal mean velocity (m/s)
    ur    = np.zeros(imax)          # radiative equilibrium u_r (m/s)
    qy    = np.zeros((imax, 4))     # -PVy (sans beta)
    psi   = np.zeros((imax, 4), dtype=np.complex64)  # streamfunction
    q     = np.zeros((imax, 4), dtype=np.complex64)  # perturbation PV

    ak = np.zeros(imax)
    bk = np.zeros(imax, dtype=np.complex64)
    ck = np.zeros(imax)
    dk = np.zeros(imax)

    ud   = np.zeros((imax, idd+1))     # u data
    epz  = np.zeros((imax, idd+1))     # EP flux data
    tusu = np.zeros((imax, idd+1))     # turning surface

    psi_time = np.zeros((imax, idd+1), dtype=np.complex64)
    q_time   = np.zeros((imax, idd+1), dtype=np.complex64)


    # diagnostics at selected levels (25, 30, 35, 40 km)
    def diag_array():
        arr = np.zeros(idd+1)
        arr[:] = np.nan
        return arr

    ubar25 = diag_array(); betae25 = diag_array(); wa25 = diag_array()
    ua25   = diag_array(); epz25   = diag_array()
    ubar30 = diag_array(); betae30 = diag_array(); wa30 = diag_array()
    ua30   = diag_array(); epz30   = diag_array()
    ubar35 = diag_array(); betae35 = diag_array(); wa35 = diag_array()
    ua35   = diag_array(); epz35   = diag_array()
    ubar40 = diag_array(); betae40 = diag_array(); wa40 = diag_array()
    ua40   = diag_array(); epz40   = diag_array()

    ur[:] = np.nan

    # -----------------------------
    # set z, alpha, ur
    # -----------------------------

    for i in range(imax):
        z[i] = dz*i + 10000.0      # altitude (10000 - 80000 m)
        if alpha_on:
            alpha[i] = (1.5 + np.tanh((z[i]-35000.0)/h0))*1.0e-6
        else:
            alpha[i] = 0.0
        ur[i] = 12.0 + 0.003*(z[i]-10000.0)

    # -----------------------------
    # Initialize u0, qy, psi, q
    # -----------------------------

    for i in range(imax):
        u0[i, :] = 12.0 + (z[i]-10000.0)*(52.0-12.0)/40000.0

    for i in range(1, imax-1):
        qy[i, :] = -l*l*u0[i,0] + (f0*f0/ensq)*(
            (u0[i+1,0] + u0[i-1,0] - 2.0*u0[i,0])/(dz*dz)
            - (u0[i+1,0] - u0[i-1,0])/(2.0*dz*h0)
        )
        psi[i, :] = 0.0 + 0.0j
        q[i,   :] = 0.0 + 0.0j

    # if verbose:
    #     print("initialized")

    # -----------------------------
    # Time stepping (Adams–Bashforth 3rd order)
    # -----------------------------

    m  = 0     # time step
    mm = 0     # daily output index

    i25 = i30 = i35 = i40 = 0

    steps_per_day = int(3600*24.0/dt)

    while m < mmax:
        t = dt * m  # current time (s)

        # ---- q, qy tend ----
        i = 1
        while i < imax-1:
            betae1 = beta - eps*qy[i,0]
            betae2 = beta - eps*qy[i,1]
            betae3 = beta - eps*qy[i,2]

            uadv1 = -(0.0+1.0j)*k*eps*u0[i,0]*q[i,0]
            uadv2 = -(0.0+1.0j)*k*eps*u0[i,1]*q[i,1]
            uadv3 = -(0.0+1.0j)*k*eps*u0[i,2]*q[i,2]

            vadv1 = -betae1*(0.0+1.0j)*k*psi[i,0]
            vadv2 = -betae2*(0.0+1.0j)*k*psi[i,1]
            vadv3 = -betae3*(0.0+1.0j)*k*psi[i,2]

            damp0 = (f0*f0/ensq)*(0.5/h0)*alpha[i]*(
                (psi[i+1,2]-psi[i-1,2])/(2.0*dz) + psi[i,2]/(2.0*h0)
            )
            damp1 = -(f0*f0/(ensq*dz*dz))*0.5*(alpha[i+1]+alpha[i])*(psi[i+1,2]-psi[i,2])
            damp2 = +(f0*f0/(ensq*dz*dz))*0.5*(alpha[i-1]+alpha[i])*(psi[i,2]-psi[i-1,2])
            damp3 = -(f0*f0/(ensq*dz*h0))*0.125
            damp3 = damp3*((alpha[i+1]+alpha[i])*psi[i+1,2]
                           -(alpha[i-1]+alpha[i])*psi[i-1,2]
                           +(alpha[i+1]-alpha[i-1])*psi[i,2])

            q[i,3] = q[i,2] + (dt/12.0)*(
                5.0*uadv1 - 16.0*uadv2 + 23.0*uadv3
                + 5.0*vadv1 - 16.0*vadv2 + 23.0*vadv3
            )
            q[i,3] = q[i,3] + dt*(damp0 + damp1 + damp2 + damp3)

            zm = 0.5*(z[i]+z[i-1])/h0
            zp = 0.5*(z[i]+z[i+1])/h0

            udamp1 = -(f0*f0/ensq)*np.exp(z[i]/h0)*(1.0/dz)*(
                0.5*(alpha[i+1]+alpha[i])*np.exp(-zp)
                * ((u0[i+1,2]-u0[i,2])/dz - urz)
            )
            udamp2 = (f0*f0/ensq)*np.exp(z[i]/h0)*(1.0/dz)*(
                0.5*(alpha[i-1]+alpha[i])*np.exp(-zm)
                * ((u0[i,2]-u0[i-1,2])/dz - urz)
            )

            flux1 = 0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(
                psi[i,0]*np.conj(psi[i+1,0]+psi[i-1,0]-2.0*psi[i,0])/(dz*dz)
            )
            flux2 = 0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(
                psi[i,1]*np.conj(psi[i+1,1]+psi[i-1,1]-2.0*psi[i,1])/(dz*dz)
            )
            flux3 = 0.5*l*l*k*eps*(f0*f0/ensq)*np.exp(z[i]/h0)*np.imag(
                psi[i,2]*np.conj(psi[i+1,2]+psi[i-1,2]-2.0*psi[i,2])/(dz*dz)
            )

            qy[i,3] = qy[i,2] + (dt/12.0)*(5.0*flux1 - 16.0*flux2 + 23.0*flux3)
            qy[i,3] = qy[i,3] + dt*(udamp1 + udamp2)
            i += 1

        # ---- invert psi from q ----
        ak[imax-2] = 0.0
        bk[imax-2] = 0.0 + 0.0j
        i = imax-2
        while i > -1:
            ak[i-1] = -1.0/(ak[i] - p)
            bk[i-1] = (r*q[i,3] - bk[i])/(ak[i] - p)
            i -= 1

        psi[0,3] = hb*(g/f0)*(1.0 - np.exp(-t/tau))   # lower BC

        i = 0
        while i < imax-1:
            psi[i+1,3] = ak[i]*psi[i,3] + bk[i]
            i += 1

        # ---- invert u0 from qy ----
        ck = ck   # just to emphasize reuse
        dk = dk

        ck[imax-2] = 1.0
        dk[imax-2] = 0.0
        i = imax-2
        while i > -1:
            ck[i-1] = -vp/(ck[i]*vm - pu)
            dk[i-1] = (r*qy[i,3] - dk[i]*vm)/(ck[i]*vm - pu)
            i -= 1

        u0[0,3] = 12.0   # lower BC: fixed u0
        i = 0
        while i < imax-1:
            u0[i+1,3] = ck[i]*u0[i,3] + dk[i]
            i += 1

        # ---- shift time levels ----
        qy[:,0] = qy[:,1]; qy[:,1] = qy[:,2]; qy[:,2] = qy[:,3]
        q[:,0]  = q[:,1];  q[:,1]  = q[:,2];  q[:,2]  = q[:,3]
        psi[:,0]= psi[:,1];psi[:,1]= psi[:,2];psi[:,2]= psi[:,3]
        u0[:,0] = u0[:,1]; u0[:,1] = u0[:,2]; u0[:,2] = u0[:,3]

        # ---- daily output ----
        if m % steps_per_day == 0:
            day = t/(3600.0*24.0)
            # if verbose:
            #     print("m =", m, " time =", day, " days  ", u0[22,2])

            ud[:,mm] = u0[:,2]
            psi_time[:, mm] = psi[:, 2]
            q_time[:, mm]   = q[:, 2]

            i = 1
            while i < imax-1:
                epz[i,mm] = -0.5*k*eps*np.imag(
                    psi[i,1]*np.conj(psi[i+1,1]-psi[i-1,1])/(2.0*dz)
                )
                tusu[i,mm] = eps*ud[i,mm]/(beta-eps*qy[i,1]) - 1.0/(k**2 + l**2 + lrmsq)
                i += 1

            # 25 km
            if i25 == 0:
                ubar25[mm]  = u0[16,1]
                betae25[mm] = beta - eps*qy[16,1]
                aa25        = abs(q[16,1])
                wa25[mm]    = eps*aa25*aa25*np.exp(25.0/7.0)*0.25/betae25[mm]
                epz25[mm]   = -0.5*k*eps*np.imag(
                    psi[16,1]*np.conj(psi[17,1]-psi[15,1])/(2.0*dz)
                )
                ua25[mm]    = u0[16,1]
                if betae25[mm] < 0:
                    i25 = mm

            # 30 km
            if i30 == 0:
                ubar30[mm]  = u0[21,1]
                betae30[mm] = beta - eps*qy[21,1]
                aa30        = abs(q[21,1])
                wa30[mm]    = eps*aa30*aa30*np.exp(30.0/7.0)*0.25/betae30[mm]
                epz30[mm]   = -0.5*k*eps*np.imag(
                    psi[21,1]*np.conj(psi[22,1]-psi[20,1])/(2.0*dz)
                )
                ua30[mm]    = u0[21,1]
                if betae30[mm] < 0:
                    i30 = mm

            # 35 km
            if i35 == 0:
                ubar35[mm]  = u0[26,1]
                betae35[mm] = beta - eps*qy[26,1]
                aa35        = abs(q[26,1])
                wa35[mm]    = eps*aa35*aa35*np.exp(35.0/7.0)*0.25/betae35[mm]
                ua35[mm]    = u0[26,1]
                epz35[mm]   = -0.5*k*eps*np.imag(
                    psi[26,1]*np.conj(psi[27,1]-psi[25,1])/(2.0*dz)
                )
                if betae35[mm] < 0:
                    i35 = mm

            # 40 km
            if i40 == 0:
                ubar40[mm]  = u0[31,1]
                betae40[mm] = beta - eps*qy[31,1]
                aa40        = abs(q[31,1])
                wa40[mm]    = eps*aa40*aa40*np.exp(40.0/7.0)*0.25/betae40[mm]
                epz40[mm]   = -0.5*k*eps*np.imag(
                    psi[31,1]*np.conj(psi[32,1]-psi[30,1])/(2.0*dz)
                )
                ua40[mm]    = u0[31,1]
                if betae40[mm] < 0:
                    i40 = mm

            mm += 1

        m += 1

    # -----------------------------
    # Scale EPZ and adjust ua*
    # -----------------------------
    c = f0*f0/(ensq*4.0e-3)
    for i in range(1, imax-1):
        epz[i, :] = 0.25*c*np.exp(z[i]/7000.0)*epz[i, :]/(ud[i,20]*ud[i,20])

    ua30[:] -= ud[21,20]   # mm = 20 is 5 days
    ua40[:] -= ud[31,20]
    ua35[:] -= ud[26,20]
    ua25[:] -= ud[16,20]

    # -----------------------------
    # pack outputs
    # -----------------------------
    out = dict(
        z=z,beta=beta,eps=eps,
        ud=ud,
        epz=epz,
        tusu=tusu,
        ubar25=ubar25, betae25=betae25, wa25=wa25, ua25=ua25, epz25=epz25,
        ubar30=ubar30, betae30=betae30, wa30=wa30, ua30=ua30, epz30=epz30,
        ubar35=ubar35, betae35=betae35, wa35=wa35, ua35=ua35, epz35=epz35,
        ubar40=ubar40, betae40=betae40, wa40=wa40, ua40=ua40, epz40=epz40,
        hb=hb, tau=tau, dt=dt, dz=dz, psi_time=psi_time,q_time=q_time
    )
    return out


# %%
import numpy as np

def inner_WA(psiA, qA, psiB, qB, beta_eff, dz, beta_min_frac=1e-3):
    """
    Wave-activity-type inner product:
    <A,B>_WA = Σ dz * Re[ (qA* psiB + qB* psiA) / (2 * beta_eff) ]

    To avoid division by zero when beta_eff ≈ 0, apply a small cutoff
    for very small |beta_eff|.
    """
    psiA = np.asarray(psiA); qA = np.asarray(qA)
    psiB = np.asarray(psiB); qB = np.asarray(qB)
    beta_eff = np.asarray(beta_eff)

    beta_safe = beta_eff.copy()
    # Use a relative minimum value of β as cutoff, preserving the sign
    beta0 = np.max(np.abs(beta_eff))
    tiny = beta_min_frac * beta0
    mask = np.abs(beta_safe) < tiny
    beta_safe[mask] = np.sign(beta_safe[mask] + 1e-12) * tiny

    # Symmetrized combination
    num = np.conj(qA) * psiB + np.conj(qB) * psiA
    val = np.real(num / beta_safe)
    return np.sum(val) * dz


# %%
def project_WA_two_modes(psi_time, q_time,
                         psi_s, q_s,
                         psi_t, q_t,
                         beta_eff, dz):
    """
    Project psi(z, t), q(z, t) onto two modes
    Phi_s = (psi_s, q_s), Phi_t = (psi_t, q_t) using the wave-activity inner product:

      G_ij = <Phi_i, Phi_j>_WA
      b_i(t) = <Psi(t), Phi_i>_WA
      a(t) solves G a = b

    Returns a_s(t), a_t(t).
    """
    psi_time = np.asarray(psi_time)
    q_time   = np.asarray(q_time)
    nz, nt   = psi_time.shape

    # ---- 1. Gram matrix G (2x2) ----
    G11 = inner_WA(psi_s, q_s, psi_s, q_s, beta_eff, dz)
    G22 = inner_WA(psi_t, q_t, psi_t, q_t, beta_eff, dz)
    G12 = inner_WA(psi_s, q_s, psi_t, q_t, beta_eff, dz)
    G21 = inner_WA(psi_t, q_t, psi_s, q_s, beta_eff, dz)

    G = np.array([[G11, G12],
                  [G21, G22]], dtype=np.float64)  # WA inner product is real

    # Check condition number to avoid near-singular matrix
    condG = np.linalg.cond(G)
    print("WA Gram matrix G =\n", G)
    print("cond(G) =", condG)

    Minv = np.linalg.inv(G)

    # ---- 2. Projection at each time step ----
    a_s = np.zeros(nt, dtype=np.float64)   # a(t) is a real “wave-activity weight”
    a_t = np.zeros(nt, dtype=np.float64)

    for j in range(nt):
        psi_j = psi_time[:, j]
        q_j   = q_time[:, j]

        b1 = inner_WA(psi_j, q_j, psi_s, q_s, beta_eff, dz)
        b2 = inner_WA(psi_j, q_j, psi_t, q_t, beta_eff, dz)

        b  = np.array([b1, b2], dtype=np.float64)
        a  = Minv @ b

        a_s[j] = a[0]
        a_t[j] = a[1]

    return dict(a_s=a_s, a_t=a_t, G=G)


# %%
hb  = 69
tau = 3600.0*24.0*3  #3600.0*24.0*2

out_full = run_hm76(hb, tau, n_days=400, alpha_on=True, verbose=True)

ud  = out_full["ud"]      # (71, 150)
# epz = out_full["epz"]
u_10hpa = ud[22, :]  # time series at 10 hPa

# %%
dz = 1000.0  # vertical grid spacing
imax = 71
a = 6378000.0        # Earth's radius
l = 3.0/a            # dimensional meridional wavenumber (1/m)
ensq = 4.0e-4
h0 = 7000.0
omega = 7.292e-5
f0 = 2.0*omega*np.sin(np.pi/3.0)
u_bg = np.zeros(imax)
qy_bg = np.zeros(imax)

for i in range(1, imax-1):
    u_bg[i] = 12.0 + (z[i]-10000.0)*(52.0-12.0)/40000.0
    qy_bg[i] = -l*l*u_bg[i] + (f0*f0/ensq)*(
        (u_bg[i+1] + u_bg[i-1] - 2.0*u_bg[i])/(dz*dz)
        - (u_bg[i+1] - u_bg[i-1])/(2.0*dz*h0)
    )

beta = out_full["beta"]
eps  = out_full["eps"]
beta_eff = beta - eps * qy_bg   # effective beta from background state

psi_full = out_full['psi_time']
q_full   = out_full['q_time']

# stationary mode
psi_s = psi_stationary              # (nz,)
q_s   = out_stat['q_time'][:, -1]   # q(z) on the last day

# transient mode
psi_t = psi_tr                      # (nz,)
q_t   = q_tr                        # (nz,)

proj_WA = project_WA_two_modes(
    psi_time=psi_full,
    q_time=q_full,
    psi_s=psi_s, q_s=q_s,
    psi_t=psi_t, q_t=q_t,
    beta_eff=beta_eff,
    dz=dz
)
a_s_WA = proj_WA['a_s']    # shape (nt,)
a_t_WA = proj_WA['a_t']


# %%
# (1) define a vertical L2 inner product,
# (2) orthonormalize two vertical modes under this inner product, and
# (3) project a time-evolving field psi(z, t) onto these two modes to obtain
#     their complex modal amplitudes as functions of time.

def vertical_inner_product_L2(a, b, dz):
    return np.vdot(a, b) * dz

def orthonormalize_two_modes_L2(psi_s, psi_t, dz):
    # 1) normalize stationary mode
    N_s = vertical_inner_product_L2(psi_s, psi_s, dz)
    phi_s = psi_s / np.sqrt(N_s)

    # 2) subtract projection of transient onto phi_s direction
    proj_ts = vertical_inner_product_L2(phi_s, psi_t, dz)
    psi_t_orth = psi_t - proj_ts * phi_s
    N_t = vertical_inner_product_L2(psi_t_orth, psi_t_orth, dz)
    phi_t = psi_t_orth / np.sqrt(N_t)
    return phi_s, phi_t

def project_L2_time_series(psi_time, phi_s, phi_t, dz):
    psi_time = np.asarray(psi_time)
    imax, nt = psi_time.shape
    a_s = np.zeros(nt, dtype=np.complex128)
    a_t = np.zeros(nt, dtype=np.complex128)
    for j in range(nt):
        psi_j = psi_time[:, j]
        a_s[j] = vertical_inner_product_L2(phi_s, psi_j, dz)
        a_t[j] = vertical_inner_product_L2(phi_t, psi_j, dz)
    return a_s, a_t


# %%
phi_s_L2, phi_t_L2 = orthonormalize_two_modes_L2(psi_s, psi_t, dz)
a_s_L2, a_t_L2 = project_L2_time_series(psi_full, phi_s_L2, phi_t_L2, dz)

# %%
import matplotlib.pyplot as plt
import numpy as np

def compare_L2_vs_WA(a_s_L2, a_t_L2,
                     a_s_WA, a_t_WA,
                     dt_days=1.0, u_t=u_10hpa,
                     title_suffix=''):
    """
    Compare L^2 and wave-activity (WA) projections in time.

    Parameters
    ----------
    a_s_L2, a_t_L2 : array_like
        Complex modal amplitudes from L^2 projection for stationary (s) and transient (t) modes.
    a_s_WA, a_t_WA : array_like
        Real WA coefficients from wave-activity projection for stationary (s) and transient (t) modes.
    dt_days : float, optional
        Time interval between samples in days (default: 1.0).
    u_t : array_like
        Time series of mean flow at a chosen level (e.g. 10 hPa).
    title_suffix : str, optional
        Extra text appended to the L^2 panel title.
    """
    t = np.arange(len(a_s_L2)) * dt_days

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')

    # ---- top-left: L^2 amplitude ----
    ax = axes[0, 0]
    ax.plot(t, np.abs(a_s_L2), label='|a_s| L^2 (stationary)')
    ax.plot(t, np.abs(a_t_L2), label='|a_t| L^2 (transient)', linestyle='--')
    ax.set_ylabel('Amplitude')
    ax.set_title('L^2 projection' + title_suffix)
    ax.grid(True, linestyle=':')
    ax.legend()

    # ---- bottom-left: L^2 phase ----
    ax = axes[1, 0]
    ax.plot(t, np.angle(a_s_L2), label='arg(a_s) L^2')
    ax.plot(t, np.angle(a_t_L2), label='arg(a_t) L^2', linestyle='--')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Phase (rad)')
    ax.grid(True, linestyle=':')
    ax.legend()

    # ---- top-right: WA amplitude ----
    ax = axes[0, 1]
    ax.plot(t, a_s_WA, label='a_s WA (stationary)')
    ax.plot(t, a_t_WA, label='a_t WA (transient)', linestyle='--')
    ax.set_ylabel('WA coefficient')
    ax.set_title('Wave-activity projection (hb=' + str(hb) +
                 ', tau=' + str(tau/3600/24) + ' days)')
    ax.grid(True, linestyle=':')
    ax.legend()

    # # ---- bottom-right: WA phase (example placeholder) ----
    # ax = axes[1,1]
    # ax.plot(t, a_s_WA + a_t_WA, label='a_s + a_t')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('sum')
    # ax.grid(True, linestyle=':')
    # ax.legend()

    # # Example: interference index from L^2 amplitudes (commented out)
    # a_s = a_s_L2   # complex
    # a_t = a_t_L2   # complex
    # nt = len(a_s_L2)
    # dt_out_days = 1.0
    # t_days = np.arange(nt) * dt_out_days
    #
    # A_s = np.abs(a_s)
    # A_t = np.abs(a_t)
    # phi_s = np.angle(a_s)
    # phi_t = np.angle(a_t)
    #
    # # Phase difference; unwrap to make it continuous
    # dphi = np.unwrap(phi_s - phi_t)
    #
    # I = A_s * A_t * np.cos(dphi)               # dimensional interference index
    # I_norm = I / (A_s**2 + A_t**2 + 1e-30)    # normalized index (avoid division by 0)
    #
    # ax = axes[1, 1]
    # ax.plot(t_days, I_norm, label='Interference index I_norm')
    # ax.axhline(0, color='k', linewidth=0.8)
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('I_norm')
    # ax.set_title('Stationary–Transient interference')
    # ax.grid(True, linestyle=':')
    # ax.legend()

    # ---- bottom-right: mean flow at target height (e.g. ~32 km) ----
    z_target = 32000  # m

    ax = axes[1, 1]
    ax.plot(t, u_t, label=f'u({z_target/1000:.0f} km)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('u (m/s)')
    ax.set_title(f'Mean flow at ~{z_target/1000:.0f} km')
    ax.grid(True, linestyle=':')
    ax.set_ylim(-40, 90)
    ax.legend()

    plt.tight_layout()
    plt.show()


# %%
compare_L2_vs_WA(a_s_L2, a_t_L2,
                 a_s_WA, a_t_WA,
                 dt_days=1.0,
                 title_suffix=' (hb='+str(hb)+', tau='+str(tau/3600/24)+' days)')


# %%
print('max |a_s_WA| =', np.max(np.abs(a_s_WA)))
print('max |a_t_WA| =', np.max(np.abs(a_t_WA)))
print('ratio max_s / max_t =',
      np.max(np.abs(a_s_WA)) / np.max(np.abs(a_t_WA)))


# %%



