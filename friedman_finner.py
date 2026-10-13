# -*- coding: utf-8 -*-
"""
Friedman test + Finner post-hoc (adjusted p-values)
Metrics: DA and gross trading metrics (return and Sharpe before transaction costs).
         Error metrics (MSE/RMSE/MAE) are covered by the Diebold-Mariano test (dm_test.py).
Input : the MEAN over seeds 42/77/123 of every model, per stock, built automatically from
          results/Baselines_mean_numeric_<S>.csv   (Naive, RW+Drift, ARIMAX, RF, XGBoost)
          results/Proposed_mean_numeric_<S>.csv    (AG-LSTM, AG-GRU, AG-LSTM-GRU)
        Deterministic models (Naive, RW+Drift, ARIMAX, XGBoost) enter with their single value.
        Buy & Hold is not a forecasting model (no DA) and is excluded.
Blocks: stocks (N = 4, low power - read the results with care).  Treatments: models (k).
        One analysis per metric (several metrics -> report them as descriptive evidence).
Output: results/Friedman_avg_ranks.csv, Friedman_summary.csv, Finner_posthoc.csv
        average ranks, Friedman chi2 + Iman-Davenport F, Finner-adjusted p-values of every model
        vs. a control model (default: the model with the best average rank, i.e. data-driven;
        set CONTROL = "AG-LSTM-GRU" for an a-priori control).
Usage:  python src/friedman_finner.py
"""
import os
import numpy as np, pandas as pd
from scipy import stats

# ---------------- settings ----------------
STOCKS  = ["MASR", "CCAP", "PHDC", "HELI"]
OUT_DIR = os.environ.get("EGX_OUT_DIR", "results")
MODELS  = None            # None = all models found in every stock; or e.g. ["ARIMAX","RF","XGBoost","AG-LSTM","AG-GRU","AG-LSTM-GRU"]
EXCLUDE = ["Buy & Hold"]
CONTROL = None            # None = best average rank for each metric; or e.g. "AG-LSTM-GRU"
ALPHA   = 0.05
METRICS = ["DA", "Ret_gross_%", "Sharpe_gross"]   # directional accuracy and gross (before-cost) trading metrics
# ------------------------------------------

def _read(name):
    return pd.read_csv(os.path.join(OUT_DIR, name), encoding="utf-8-sig").set_index("Model")

def load():
    tabs = {}
    for st in STOCKS:
        try:
            d = pd.concat([_read(f"Baselines_mean_numeric_{st}.csv"),
                           _read(f"Proposed_mean_numeric_{st}.csv")])
            d = d.drop(index=[m for m in EXCLUDE if m in d.index]).apply(pd.to_numeric, errors="coerce")
            tabs[st] = d
        except FileNotFoundError as e:
            print(f"[skip] {st}: {e.filename} not found (run baselines.py and proposed_mean_std.py)")
    return tabs

def finner_adjust(p):
    """Finner step-down: p_(1)<=...<=p_(m);  adj_(i) = max_{j<=i} 1-(1-p_(j))^(m/j)."""
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order, start=1):
        val = 1.0 - (1.0 - p[idx]) ** (m / rank)
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj

def analyse(tabs, metric, models):
    N = len(tabs); k = len(models)
    M = np.array([[tabs[st].loc[m, metric] for m in models] for st in tabs], float)   # N x k
    rk = np.array([stats.rankdata(-r) for r in M])  # rank 1 = best
    avg = rk.mean(axis=0)

    # Friedman (tie-corrected, scipy) + Iman-Davenport
    chi2, p_f = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    F = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
    p_F = 1 - stats.f.cdf(F, k - 1, (k - 1) * (N - 1)) if np.isfinite(F) else 0.0

    ctrl = models[int(np.argmin(avg))] if CONTROL is None else CONTROL
    ci = models.index(ctrl)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    others = [j for j in range(k) if j != ci]
    z  = np.array([(avg[j] - avg[ci]) / se for j in others])
    p_raw = 2 * (1 - stats.norm.cdf(np.abs(z)))
    p_adj = finner_adjust(p_raw)

    ranks_df = pd.DataFrame({"Model": models, "AvgRank": avg}).sort_values("AvgRank")
    post = pd.DataFrame({"Metric": metric, "Control": ctrl,
                         "Model": [models[j] for j in others],
                         "AvgRank_model": [avg[j] for j in others],
                         "AvgRank_control": avg[ci],
                         "z": z, "p_raw": p_raw, "p_Finner": p_adj,
                         "Significant": p_adj < ALPHA})
    summ = {"Metric": metric, "N_stocks": N, "k_models": k,
            "Friedman_chi2": chi2, "Friedman_p": p_f,
            "ImanDavenport_F": F, "ImanDavenport_p": p_F, "Control": ctrl}
    return ranks_df, post, summ

if __name__ == "__main__":
    tabs = load()
    if len(tabs) < 2:
        raise SystemExit("Need at least 2 stock files.")
    common = set.intersection(*[set(d.index) for d in tabs.values()])
    models = [m for m in (MODELS or list(next(iter(tabs.values())).index)) if m in common]
    print(f"Stocks (N={len(tabs)}): {list(tabs)} | Models (k={len(models)}): {models}")
    if len(tabs) < 5:
        print("NOTE: with few stocks the test has low power; read the results with care.")

    all_ranks, all_post, all_sum = [], [], []
    for met in METRICS:
        if not all(met in d.columns for d in tabs.values()):
            continue
        r, post, summ = analyse(tabs, met, models)
        r.insert(0, "Metric", met)
        all_ranks.append(r); all_post.append(post); all_sum.append(summ)
        print("\n" + "=" * 72)
        print(f"{met}  (higher is better)")
        print(f"  Friedman chi2={summ['Friedman_chi2']:.3f}, p={summ['Friedman_p']:.4f} | "
              f"Iman-Davenport F={summ['ImanDavenport_F']:.3f}, p={summ['ImanDavenport_p']:.4f}")
        print("  Average ranks:", ", ".join(f"{a}={b:.2f}" for a, b in zip(r['Model'], r['AvgRank'])))
        print(f"  Finner post-hoc vs control = {summ['Control']}")
        print(post[["Model", "AvgRank_model", "z", "p_raw", "p_Finner", "Significant"]]
              .round(4).to_string(index=False))

    pd.concat(all_ranks).round(4).to_csv(os.path.join(OUT_DIR, "Friedman_avg_ranks.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_sum).round(6).to_csv(os.path.join(OUT_DIR, "Friedman_summary.csv"), index=False, encoding="utf-8-sig")
    pd.concat(all_post).round(6).to_csv(os.path.join(OUT_DIR, "Finner_posthoc.csv"), index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUT_DIR}/: Friedman_avg_ranks.csv, Friedman_summary.csv, Finner_posthoc.csv")
