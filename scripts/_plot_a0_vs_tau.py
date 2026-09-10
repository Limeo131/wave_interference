import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

d = np.load('output/data/hm_model/step2_tau_sweep.npz')
tau = d['tau_values']
a0_steady = d['a0_steady']
psi_stat_max = d['psi_stat_max']
T_mode0 = float(d['T_mode0'])

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# Panel (a): |a_0| vs tau on log-log
ax = axes[0]
ax.plot(tau, a0_steady / 1e6, 'C0o-', lw=2.5, ms=7, label='Steady-state |a0| (RMS)')
tau_ref = np.linspace(tau[0], tau[-1], 100)
a0_at_10 = np.interp(10, tau, a0_steady)
a0_theory = a0_at_10 * (10.0 / tau_ref)
ax.plot(tau_ref, a0_theory / 1e6, 'k--', lw=1.5, alpha=0.5,
        label=r'$\propto 1/\tau$ reference')
ax.set_xlabel(r'Forcing spin-up time $\tau$ (days)')
ax.set_ylabel(r'Mode-0 amplitude $|a_0|$ ($\times 10^6$ m$^2$/s)')
ax.set_title(r'(a)  Free-mode excitation vs $\tau$ (no-WMFI, $h_b=30$ m)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, ls=':', alpha=0.4, which='both')
ax.legend(loc='upper right')

# Panel (b): ratio |a_0|/|psi_stat| vs tau
ax = axes[1]
ratio = a0_steady / psi_stat_max
ax.plot(tau, ratio, 'C3s-', lw=2.5, ms=7)
ax.set_xlabel(r'Forcing spin-up time $\tau$ (days)')
ax.set_ylabel(r'$|a_0| \,/\, |\psi_{\mathrm{stat}}|$')
ax.set_title('(b)  Fractional free-mode excitation relative to stationary wave')
ax.set_xscale('log')
ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.grid(True, ls=':', alpha=0.4, which='both')
ax.set_ylim(0, 1.1)

plt.tight_layout()
save_path = 'output/figures/hm_model/step2/step2_a0_vs_tau.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path}')
