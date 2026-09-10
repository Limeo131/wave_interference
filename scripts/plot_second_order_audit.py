"""
scripts/plot_second_order_audit.py
==================================
STEPS 7 (+10): diagnostic/audit figures for the second-order boundary test.
Audit styling only -- does NOT touch manuscript/figures/ or Fig 6.

Figures (PNG + PDF) written to:
    output/figures/hm_model/second_order/

  fig_boundary_comparison.*   : (hb,tau) plane -- actual transition boundary,
                                Mmax=0 boundary, and predicted boundaries
                                (all 4 windows).
  fig_tau_vs_hbc.*            : tau vs hb_c line comparison (easier to read).
  fig_secular_growth.*        : u2 layer amplitude vs time for rep taus (window
                                sensitivity justification).
  fig_representative_tau{10,20,50}.* : per-tau diagnostics at predicted hbc:
                                u0+hb^2 u2, Ucrit, Ucrit-u, min-over-layer(t).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.second_order_boundary as so

DATA = os.path.join(ROOT, "output", "data", "hm_model")
FIGDIR = os.path.join(ROOT, "output", "figures", "hm_model", "second_order")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"})

sob = np.load(os.path.join(DATA, "second_order_boundary.npz"), allow_pickle=True)
ref = np.load(os.path.join(DATA, "audit_reference_boundaries.npz"))

tau = sob["tau_values"]
WINDOWS = list(sob["windows"])
btau = ref["boundary_transition_tau"]; bhb = ref["boundary_transition_hb"]
mtau = ref["boundary_Mmax0_tau"]; mhb = ref["boundary_Mmax0_hb"]

# per-tau actual/Mmax0 aligned to tau grid
act = np.array([bhb[np.where(np.isclose(btau, t))[0][0]] if np.any(np.isclose(btau, t))
                else np.nan for t in tau])
mm0 = np.array([mhb[np.where(np.isclose(mtau, t))[0][0]] if np.any(np.isclose(mtau, t))
                else np.nan for t in tau])


def _save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)
    print("  saved", name)


# ── Fig 1: boundary comparison in (hb, tau) plane ────────────────────────────
def fig_boundary_comparison():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(act, tau, "k--", lw=2.0, label="Actual transition boundary (nonlinear)")
    ax.plot(mm0, tau, color="0.5", lw=1.6, ls="-", label=r"Nonlinear $M_\mathrm{max}=0$")
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(WINDOWS)))
    for c, w in zip(cmap, WINDOWS):
        p = sob[f"hbc_pred_w{w}"]
        ax.plot(p, tau, "-", color=c, lw=1.5,
                label=rf"2nd-order pred. ({w} d window)")
    ax.set_xlabel(r"$h_b$ (m)")
    ax.set_ylabel(r"$\tau$ (days)")
    ax.set_title("Second-order predicted vs nonlinear transition boundary")
    ax.set_xlim(15, 62); ax.set_ylim(0, 62)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, "fig_boundary_comparison")


# ── Fig 2: tau vs hb_c line comparison ───────────────────────────────────────
def fig_tau_vs_hbc():
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(tau, act, "k--", lw=2.0, label="Actual transition")
    ax.plot(tau, mm0, color="0.5", lw=1.6, label=r"Nonlinear $M_\mathrm{max}=0$")
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(WINDOWS)))
    for c, w in zip(cmap, WINDOWS):
        ax.plot(tau, sob[f"hbc_pred_w{w}"], "-", color=c, lw=1.4,
                label=rf"2nd-order ({w} d)")
    ax.set_xlabel(r"$\tau$ (days)"); ax.set_ylabel(r"$h_{b,c}$ (m)")
    ax.set_title(r"Critical forcing amplitude $h_{b,c}(\tau)$")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    _save(fig, "fig_tau_vs_hbc")


# ── Fig 3: secular-growth diagnostic ─────────────────────────────────────────
def fig_secular_growth():
    amp = sob["u2_layer_amp"]  # (ntau, ndays) per hb^2
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    days = np.arange(amp.shape[1])
    for ti, lab in [(9, r"$\tau=10$ d"), (19, r"$\tau=20$ d"), (49, r"$\tau=50$ d")]:
        ax.plot(days, amp[ti], lw=1.2, label=lab)
    for w in WINDOWS:
        ax.axvline(w, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("day"); ax.set_ylabel(r"$\max_{25-40\,\mathrm{km}}|u_2|/h_b^2$")
    ax.set_title("One-way second-order response amplitude (no secular runaway)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _save(fig, "fig_secular_growth")


# ── Fig 4: representative-tau diagnostics ────────────────────────────────────
def fig_representative(tau_val):
    key_u2 = f"u2_time_tau{int(tau_val)}"
    key_G2 = f"G2_tau{int(tau_val)}"
    if key_u2 not in sob.files:
        print("  (no rep data for tau", tau_val, ")"); return
    u2_time = sob[key_u2]
    G2 = sob[key_G2]
    G0 = sob["G0"]
    Z = sob["Z"]; U_BG = sob["U_BG"]
    zkm = Z / 1000.0
    zmask = (zkm >= 25.0) & (zkm <= 40.0)
    # use the predicted hbc (1000-day window) for this tau
    ti = np.where(np.isclose(tau, tau_val))[0][0]
    hbc = float(sob["hbc_pred_w1000"][ti])
    hb = hbc
    u_pred = U_BG[:, None] + hb * hb * u2_time
    D_pred = G0[:, None] + hb * hb * G2
    # Ucrit(u_pred) exact and Ucrit0
    Ucrit_pred = so.Dmargin_exact(u_pred) + u_pred  # = Ucrit(u_pred)
    _, Ucrit0, _ = so.G0_profile()

    # min over layer vs time (predicted margin)
    minlayer = np.min(D_pred[zmask, :], axis=0)
    topen = np.where(minlayer >= 0)[0]
    t0 = int(topen[0]) if len(topen) else int(np.argmax(minlayer))
    zidx = np.where(zmask)[0]
    zloc = zidx[np.argmin(D_pred[zmask, t0])]

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.0))
    fig.suptitle(rf"Representative diagnostics $\tau={tau_val:.0f}$ d, "
                 rf"$h_b=h_{{b,c}}^{{\rm pred}}={hbc:.1f}$ m "
                 rf"(actual boundary $\approx{act[ti]:.1f}$ m)", fontsize=11)

    # (a) winds at t0
    ax = axs[0, 0]
    ax.plot(U_BG, zkm, "k-", lw=1.5, label=r"$u_0$")
    ax.plot(u_pred[:, t0], zkm, "b-", lw=1.5, label=r"$u_0+h_b^2 u_2$ (day %d)" % t0)
    ax.plot(Ucrit0, zkm, "g--", lw=1.2, label=r"$U_\mathrm{crit}(u_0)$")
    ax.plot(Ucrit_pred[:, t0], zkm, "r--", lw=1.2, label=r"$U_\mathrm{crit}(u_\mathrm{pred})$")
    ax.axhspan(25, 40, color="orange", alpha=0.12)
    ax.set_xlabel("wind (m/s)"); ax.set_ylabel("z (km)"); ax.set_ylim(10, 55)
    ax.legend(fontsize=7.5); ax.set_title("(a) winds & critical wind")
    ax.grid(alpha=0.25)

    # (b) margin Ucrit - u at t0
    ax = axs[0, 1]
    ax.plot(G0, zkm, "g-", lw=1.2, label=r"$G_0=U_\mathrm{crit}(u_0)-u_0$")
    ax.plot(D_pred[:, t0], zkm, "r-", lw=1.5, label=r"$D_\mathrm{pred}$ (day %d)" % t0)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhspan(25, 40, color="orange", alpha=0.12)
    ax.axhline(Z[zloc] / 1000, color="0.4", ls=":", lw=0.9)
    ax.set_xlabel(r"$U_\mathrm{crit}-u$ (m/s)"); ax.set_ylabel("z (km)"); ax.set_ylim(10, 55)
    ax.legend(fontsize=7.5); ax.set_title("(b) predicted propagation margin")
    ax.grid(alpha=0.25)

    # (c) min over 25-40 km vs time
    ax = axs[1, 0]
    days = np.arange(D_pred.shape[1])
    ax.plot(days, minlayer, "b-", lw=1.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(t0, color="r", ls=":", lw=1.0, label="first opening day %d" % t0 if len(topen) else "max-open day %d" % t0)
    ax.set_xlabel("day"); ax.set_ylabel(r"$\min_{25-40\,\mathrm{km}} D_\mathrm{pred}$ (m/s)")
    ax.set_title("(c) layer-minimum margin vs time")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlim(0, 1000)

    # (d) Hovmoller of D_pred in layer (z-t)
    ax = axs[1, 1]
    nt = D_pred.shape[1]
    ex = [0, nt, 25, 40]
    im = ax.imshow(D_pred[zmask, :], aspect="auto", origin="lower",
                   extent=ex, cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(D_pred[zmask, :])),
                   vmax=np.nanmax(np.abs(D_pred[zmask, :])))
    cs = ax.contour(np.linspace(0, nt, nt), np.linspace(25, 40, zmask.sum()),
                    D_pred[zmask, :], levels=[0], colors="k", linewidths=1.0)
    ax.set_xlabel("day"); ax.set_ylabel("z (km)")
    ax.set_title(r"(d) $D_\mathrm{pred}(z,t)$ in barrier layer")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$D_\mathrm{pred}$ (m/s)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"fig_representative_tau{int(tau_val)}")

    return dict(tau=tau_val, hbc=hbc, t0=t0, zopen=Z[zloc] / 1000,
                minlayer_max=float(np.max(minlayer)),
                opens=len(topen) > 0)


if __name__ == "__main__":
    print("Writing audit figures to", FIGDIR)
    fig_boundary_comparison()
    fig_tau_vs_hbc()
    fig_secular_growth()
    reps = []
    for tv in [10.0, 20.0, 50.0]:
        r = fig_representative(tv)
        if r: reps.append(r)
    print("\nRepresentative-case summary:")
    for r in reps:
        print(f"  tau={r['tau']:.0f}: hbc_pred={r['hbc']:.1f} m, "
              f"barrier {'opens' if r['opens'] else 'peaks'} day {r['t0']}, "
              f"z={r['zopen']:.0f} km, max layer-min margin={r['minlayer_max']:+.3f}")
    print("Done.")
