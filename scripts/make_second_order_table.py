"""
scripts/make_second_order_table.py
==================================
STEP 8: machine-readable comparison table + error metrics.

Writes:
  output/data/hm_model/second_order_boundary_comparison.csv
      one row per tau: actual hbc, Mmax0 hbc, predicted hbc (4 windows),
      and prediction errors (pred - actual) for each window.
  output/data/hm_model/second_order_error_summary.csv
      MAE / RMS / median-abs / within{1,2,5}m / shape correlation
      per window, vs the actual transition boundary and vs Mmax0.
"""
import os, sys, csv
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "output", "data", "hm_model")

sob = np.load(os.path.join(DATA, "second_order_boundary.npz"), allow_pickle=True)
ref = np.load(os.path.join(DATA, "audit_reference_boundaries.npz"))

tau = sob["tau_values"]
WINDOWS = [int(w) for w in sob["windows"]]

btau = ref["boundary_transition_tau"]; bhb = ref["boundary_transition_hb"]
mtau = ref["boundary_Mmax0_tau"]; mhb = ref["boundary_Mmax0_hb"]
act = np.array([bhb[np.where(np.isclose(btau, t))[0][0]] if np.any(np.isclose(btau, t))
                else np.nan for t in tau])
mm0 = np.array([mhb[np.where(np.isclose(mtau, t))[0][0]] if np.any(np.isclose(mtau, t))
                else np.nan for t in tau])

pred = {w: sob[f"hbc_pred_w{w}"] for w in WINDOWS}

# ── per-tau table ────────────────────────────────────────────────────────────
csv1 = os.path.join(DATA, "second_order_boundary_comparison.csv")
with open(csv1, "w", newline="") as f:
    wr = csv.writer(f)
    header = ["tau_days", "hbc_actual_transition_m", "hbc_Mmax0_m"]
    for w in WINDOWS:
        header.append(f"hbc_pred_w{w}_m")
    for w in WINDOWS:
        header.append(f"err_pred_w{w}_minus_actual_m")
    wr.writerow(header)
    for i, t in enumerate(tau):
        row = [f"{t:.0f}", f"{act[i]:.3f}", f"{mm0[i]:.3f}"]
        for w in WINDOWS:
            row.append(f"{pred[w][i]:.3f}")
        for w in WINDOWS:
            row.append(f"{pred[w][i]-act[i]:.3f}")
        wr.writerow(row)
print("Wrote", csv1)


# ── error summary ────────────────────────────────────────────────────────────
def metrics(p, target):
    e = p - target
    fin = np.isfinite(e)
    e = e[fin]
    return dict(
        MAE=float(np.mean(np.abs(e))),
        RMS=float(np.sqrt(np.mean(e ** 2))),
        median_abs=float(np.median(np.abs(e))),
        bias=float(np.mean(e)),
        within1=float(np.mean(np.abs(e) <= 1.0)),
        within2=float(np.mean(np.abs(e) <= 2.0)),
        within5=float(np.mean(np.abs(e) <= 5.0)),
        pearson=float(pearsonr(target[fin], p[fin])[0]),
        spearman=float(spearmanr(target[fin], p[fin])[0]),
        ratio_min=float(np.min(p[fin] / target[fin])),
        ratio_max=float(np.max(p[fin] / target[fin])),
        ratio_mean=float(np.mean(p[fin] / target[fin])),
    )


csv2 = os.path.join(DATA, "second_order_error_summary.csv")
fields = ["reference", "window_days", "MAE_m", "RMS_m", "median_abs_m", "bias_m",
          "frac_within_1m", "frac_within_2m", "frac_within_5m",
          "pearson_r", "spearman_r", "ratio_min", "ratio_max", "ratio_mean"]
with open(csv2, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(fields)
    for target_name, target in [("actual_transition", act), ("Mmax0", mm0)]:
        for w in WINDOWS:
            m = metrics(pred[w], target)
            wr.writerow([target_name, w,
                         f"{m['MAE']:.3f}", f"{m['RMS']:.3f}", f"{m['median_abs']:.3f}",
                         f"{m['bias']:.3f}", f"{m['within1']:.3f}", f"{m['within2']:.3f}",
                         f"{m['within5']:.3f}", f"{m['pearson']:.4f}", f"{m['spearman']:.4f}",
                         f"{m['ratio_min']:.3f}", f"{m['ratio_max']:.3f}", f"{m['ratio_mean']:.3f}"])
print("Wrote", csv2)

print("\n=== Error summary vs ACTUAL transition boundary ===")
print(f"{'win':>5} {'MAE':>6} {'RMS':>6} {'med':>6} {'bias':>7} {'<1m':>5} {'<2m':>5} {'<5m':>5} {'r':>6} {'ratio':>12}")
for w in WINDOWS:
    m = metrics(pred[w], act)
    print(f"{w:5d} {m['MAE']:6.2f} {m['RMS']:6.2f} {m['median_abs']:6.2f} {m['bias']:7.2f} "
          f"{m['within1']:5.2f} {m['within2']:5.2f} {m['within5']:5.2f} {m['pearson']:6.3f} "
          f"{m['ratio_min']:.2f}-{m['ratio_max']:.2f}")

# also: correlation of shape after removing the mean offset (does it capture SHAPE?)
print("\n=== Shape capture (detrended) vs actual ===")
for w in WINDOWS:
    p = pred[w]; fin = np.isfinite(p) & np.isfinite(act)
    # affine fit pred = a*actual + b
    A = np.polyfit(act[fin], p[fin], 1)
    resid = p[fin] - (A[0] * act[fin] + A[1])
    rms_resid = np.sqrt(np.mean(resid ** 2))
    print(f"  w{w}: pred = {A[0]:.3f}*actual + {A[1]:+.2f} ; RMS residual after affine = {rms_resid:.3f} m")
print("Done.")
