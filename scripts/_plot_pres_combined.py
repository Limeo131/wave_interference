"""Generate the combined 4-panel presentation figure with correct LaTeX rendering."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from models.hm76 import (run_hm76, solve_hm76_eigenmodes, default_hm76_background,
                          hm76_constants, compute_betae_for_eigen)

# === Setup ===
const = hm76_constants(s=2.0, dz=1000.0)
k = const['k']; l = const['l']; eps = const['eps']; f0 = const['f0']
ensq = const['ensq']; h0 = const['h0']; dz = 1000.0
g = 9.81; imax = 71
lrmsq = f0**2 / (4*ensq*h0**2)
r = ensq * dz**2 / f0**2
p = 2.0 + r * (k**2 + l**2 + lrmsq)
z = np.arange(imax) * dz + 10000.0
z_km = z / 1000.0
u0 = default_hm76_background(z)
betae = np.real(compute_betae_for_eigen(z, u0, const))
HB = 30.0
psi_bc = HB * g / f0

# Iterative psi_stat
psi = psi_bc * (1 - np.arange(imax)/(imax-1))
for _ in range(300):
    q = np.zeros(imax)
    for i in range(1, imax-1):
        q[i] = -betae[i]*psi[i]/(eps*u0[i])
    ak = np.zeros(imax); bk = np.zeros(imax)
    for i in range(imax-2, 0, -1):
        ak[i-1] = -1.0/(ak[i]-p)
        bk[i-1] = (r*q[i]-bk[i])/(ak[i]-p)
    psi_new = np.zeros(imax); psi_new[0] = psi_bc
    for i in range(imax-1):
        psi_new[i+1] = ak[i]*psi_new[i]+bk[i]
    if np.max(np.abs(psi_new-psi)) < 1e-10:
        break
    psi = psi_new.copy()
psi_stat = psi.copy()

# Eigenmodes
eig_out = solve_hm76_eigenmodes(s=2.0, alpha_on=False, u_bg=u0, n_print=0)
order = eig_out['order']; modes = eig_out['modes']; omega = eig_out['omega']
N_mat = np.real(eig_out['Nmat'])
j0 = order[0]; phi0 = np.real(modes[:, j0])
phi0_int = phi0[1:-1]; norm_N0 = phi0_int @ N_mat @ phi0_int
N_phi0 = N_mat @ phi0_int
omega0 = np.real(omega[j0]); T0 = 2*np.pi/np.abs(omega0)/86400.0

K2 = k**2 + l**2 + const['LD_inv2']
m2 = (ensq/f0**2)*(betae/(eps*u0) - K2)
z_turn = None
for i in range(len(z)-1):
    if m2[i]*m2[i+1] < 0:
        z_turn = z_km[i]+(z_km[i+1]-z_km[i])*(-m2[i])/(m2[i+1]-m2[i])
        break

# Run no-WMFI
out = run_hm76(30.0, 10.0*86400, s=2.0, n_days=300, alpha_on=False, wave_mean_feedback=False)
psi_time = out['psi_time']; nt = psi_time.shape[1]
t = np.arange(nt, dtype=float)
tau_days = 10.0; ramp = 1.0 - np.exp(-t/tau_days)
psi_stat_t = psi_stat[:, np.newaxis] * ramp[np.newaxis, :]
psi_res = psi_time - psi_stat_t
a0 = np.array([(N_phi0 @ psi_res[1:-1, tt])/norm_N0 for tt in range(nt)])
psi_recon0 = phi0[:, np.newaxis] * a0[np.newaxis, :]

window = slice(100, 250)
var_res = np.sum(np.abs(psi_res[:, window])**2)
unexpl = np.sum(np.abs(psi_res[:, window]-psi_recon0[:, window])**2)
frac = 1.0 - unexpl/var_res
std_mean = np.std(np.abs(a0[window]))/np.mean(np.abs(a0[window]))

# Interference at 25 km
iz25 = np.argmin(np.abs(z_km-25))
psi_stat_25 = psi_stat[iz25]
psi_free_25 = a0 * phi0[iz25]
psi_total_25 = psi_stat_25 + psi_free_25
amp = np.abs(psi_total_25)

maxima = argrelextrema(amp[100:200], np.greater, order=10)[0]+100
minima = argrelextrema(amp[100:200], np.less, order=10)[0]+100
t_con = int(maxima[0]); t_des = int(minima[0])
t_q1 = (t_con+t_des)//2; t_q2 = t_des + (t_des-t_con)

# ================================================================
# 4-PANEL FIGURE
# ================================================================
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) Propagation cavity
ax = axes[0, 0]
ax.plot(m2*1e8, z_km, 'k', lw=2.5)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.fill_betweenx(z_km, 0, np.where(m2 > 0, m2*1e8, 0), alpha=0.25, color='C0')
ax.fill_betweenx(z_km, 0, np.where(m2 < 0, m2*1e8, 0), alpha=0.15, color='C3')
ax.axhline(z_turn, color='green', lw=2.5, ls='--')
phi0_sc = np.abs(phi0)/np.max(np.abs(phi0))*2.0
psi_sc = psi_stat/np.max(psi_stat)*2.0
ax.plot(phi0_sc, z_km, 'C3', lw=2.5, label='Mode 0 (T=%.1f d)' % T0)
ax.plot(psi_sc, z_km, 'C0', lw=2, ls=':', label='Forced stationary wave')
ax.text(-2.3, z_turn+1.5, 'Turning level\n%.1f km' % z_turn,
        color='green', fontsize=10, fontweight='bold')
ax.set_xlabel(r'$m^2$ ($10^{-8}$ m$^{-2}$) / normalized amplitude')
ax.set_ylabel('Height (km)')
ax.set_title('(a)  Propagation cavity, Mode 0, and forced stationary wave')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-2.5, 3); ax.set_ylim(10, 65)
ax.grid(True, ls=':', alpha=0.3)

# (b) Projection coefficient
ax = axes[0, 1]
ax.plot(t, np.real(a0)/1e6, 'C0', lw=1.2, label=r'Re($a_0$)')
ax.plot(t, np.imag(a0)/1e6, 'C1', lw=1.2, ls='--', label=r'Im($a_0$)')
ax.plot(t, np.abs(a0)/1e6, 'k', lw=2, label=r'$|a_0|$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$a_0(t)$ ($\times 10^6$ m$^2$/s)')
ax.set_title('(b)  Mode-0 projection coefficient (N-weighted)')
ax.legend(loc='upper right', ncol=3, fontsize=9)
ax.grid(True, ls=':', alpha=0.3)
info_text = ('$T_0$ = %.1f d\n$|a_0|$ std/mean = %.1f%%\n'
             'Variance expl. = %.2f%%') % (T0, std_mean*100, frac*100)
ax.text(0.02, 0.05, info_text, transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# (c) Complex-plane interference
ax = axes[1, 0]
scale = 1e6
ps = psi_stat_25/scale
ax.annotate('', xy=(ps, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='k'))
ax.text(ps*0.5, -0.4, r'$\psi_{\mathrm{stat}}$',
        fontsize=13, ha='center', color='k', fontweight='bold')

times = [t_con, t_q1, t_des, t_q2]
labels_vec = ['Constructive', 'Quadrature', 'Destructive', 'Quadrature']
colors_vec = ['C0', 'C2', 'C3', 'C4']
for day, label, col in zip(times, labels_vec, colors_vec):
    pt = psi_total_25[day]/scale
    ax.annotate('', xy=(np.real(pt), np.imag(pt)), xytext=(ps, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color=col, alpha=0.8))
    ax.plot(np.real(pt), np.imag(pt), 'o', color=col, ms=8, zorder=5)
    ax.text(np.real(pt)+0.1, np.imag(pt)+0.2, '%s\nday %d' % (label, day),
            fontsize=8, color=col, ha='center')

r_free = np.mean(np.abs(psi_free_25[100:200]))/scale
theta_c = np.linspace(0, 2*np.pi, 100)
ax.plot(ps+r_free*np.cos(theta_c), r_free*np.sin(theta_c),
        'gray', lw=1, ls='--', alpha=0.4)
ax.set_xlabel(r'Re($\psi$) ($\times 10^6$ m$^2$/s)')
ax.set_ylabel(r'Im($\psi$) ($\times 10^6$ m$^2$/s)')
ax.set_title('(c)  Complex-plane interference at 25 km')
ax.set_aspect('equal')
ax.grid(True, ls=':', alpha=0.3)
ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)

# (d) Wave-amplitude vacillation
ax = axes[1, 1]
ax.plot(t, amp/1e6, 'C0', lw=2, label=r'$|\psi_{\mathrm{total}}|$ at 25 km')
ax.axhline(psi_stat_25/1e6, color='k', lw=1.5, ls='--', alpha=0.7,
           label=r'$|\psi_{\mathrm{stat}}|$')
ax.set_xlim(80, 220)
for day, label, col in zip(times, labels_vec, colors_vec):
    ax.axvline(day, color=col, lw=1.5, ls=':', alpha=0.7)
    ax.plot(day, amp[day]/1e6, 'o', color=col, ms=8, zorder=5)
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$|\psi|$ ($\times 10^6$ m$^2$/s)')
ax.set_title('(d)  Wave-amplitude vacillation from stationary + free-mode interference')
ax.legend(loc='upper right')
ax.grid(True, ls=':', alpha=0.3)

fig.suptitle(
    r'Free-mode interference mechanism (no-WMFI, frozen day-0 $\bar{u}$,'
    r' $h_b$=30 m, $\tau$=10 d)',
    fontsize=13, y=0.995)
plt.savefig('output/figures/hm_model/step2/pres_combined_4panel.png',
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: output/figures/hm_model/step2/pres_combined_4panel.png')
