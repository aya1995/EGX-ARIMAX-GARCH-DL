# -*- coding: utf-8 -*-
"""
Diebold-Mariano test from saved forecasts (no retraining).

Reference model = AG-LSTM-GRU (the proposed model, fixed a priori), compared with every other model:
  baselines  : Naive, RW+Drift, ARIMAX, RF, XGBoost          (results/Predictions_<S>_baselines_seed<N>.csv)
  proposed   : AG-LSTM, AG-GRU                                (results/Predictions_<S>_proposed_seed<N>.csv)
  ablation   : ARIMAX+DL-LSTM / -GRU / -LSTM-GRU (no GARCH)   (results/Predictions_<S>_ablation_seed<N>.csv, optional)
Set REFERENCE = None to use the best model of friedman_finner.py instead (data-driven).

Two loss functions: squared error (RMSE-based) and absolute error (MAE-based); h = 1,
Harvey-Leybourne-Newbold small-sample correction.
Sign convention: DM < 0  =>  the reference model has the LOWER loss than the compared model.
With several seeds the script reports (i) the DM test of every seed and (ii) a seed-averaged DM:
the loss differential d_t is averaged over the seeds at each time point and the DM test is applied
to that averaged series.  Many pairwise tests are run, so read isolated p-values near 0.05 with care.

Usage:   python src/dm_test.py
Output:  results/DM_results.csv  (long format)   results/DM_summary_seed_averaged.csv  (wide, p-values)
"""
import os
import numpy as np, pandas as pd
from scipy import stats

STOCKS    = ["MASR", "CCAP", "PHDC", "HELI"]
SEEDS     = [42, 77, 123]
REFERENCE = "AG-LSTM-GRU"      # None = best average Friedman rank (needs results/Friedman_avg_ranks.csv)
OUT_DIR   = os.environ.get("EGX_OUT_DIR", "results")
SOURCES   = ["baselines", "proposed", "ablation"]      # prediction files merged per seed

LOSSES = {"RMSE": lambda e: e ** 2,        # squared-error loss (RMSE-based)
          "MAE":  lambda e: np.abs(e)}     # absolute-error loss (MAE-based)


def dm_from_d(d):
    """DM statistic and two-sided p-value (t distribution, n-1 df) for a loss differential series."""
    d = np.asarray(d, float)
    n = len(d); dbar = d.mean()
    var = np.var(d, ddof=0) / n            # h = 1 -> only gamma_0
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var) * np.sqrt((n - 1) / n)      # HLN correction (h = 1)
    return dm, 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))


def load_predictions(stock, seed):
    """Merge the forecast files of one stock and seed; returns (actual, {model: forecast})."""
    actual, preds = None, {}
    for src in SOURCES:
        f = os.path.join(OUT_DIR, f"Predictions_{stock}_{src}_seed{seed}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if actual is None:
            actual, dates = d["Actual"].values, d["Date"].values
        elif not (np.allclose(actual, d["Actual"].values) and (dates == d["Date"].values).all()):
            raise ValueError(f"{f}: dates/actual do not match the other prediction files")
        for c in d.columns:
            if c not in ("Date", "Actual") and c not in preds:     # first source wins (ARIMAX from baselines)
                preds[c] = d[c].values
    return actual, preds


if __name__ == "__main__":
    ref = REFERENCE
    if ref is None:
        r = pd.read_csv(os.path.join(OUT_DIR, "Friedman_avg_ranks.csv"), encoding="utf-8-sig")
        mean_rank = r.groupby("Model")["AvgRank"].mean().sort_values()
        ref = mean_rank.index[0]
        print("Mean Friedman rank over metrics:\n", mean_rank.round(3).to_string())
    print(f"\nReference model: {ref}\n")

    rows = []
    for st in STOCKS:
        data = {}
        for sd in SEEDS:
            a, p = load_predictions(st, sd)
            if a is not None:
                data[sd] = (a, p)
        if len(data) < len(SEEDS) or any(ref not in p for _, p in data.values()):
            print(f"[skip] {st}: forecasts of all seeds (and of {ref}) are required"); continue
        a0, p0 = data[SEEDS[0]]
        for ot in [m for m in p0 if m != ref]:
            if any(ot not in p for _, p in data.values()):
                continue
            for lname, loss in LOSSES.items():
                d_seeds = []
                for sd in SEEDS:
                    a, p = data[sd]
                    d_s = loss(a - p[ref]) - loss(a - p[ot])
                    d_seeds.append(d_s)
                    dm, pv = dm_from_d(d_s)
                    rows.append({"Stock": st, "Loss": lname, "Reference": ref, "Compared_with": ot,
                                 "Seed": sd, "DM": round(dm, 4), "p_value": round(pv, 4)})
                dm, pv = dm_from_d(np.mean(d_seeds, axis=0))
                rows.append({"Stock": st, "Loss": lname, "Reference": ref, "Compared_with": ot,
                             "Seed": "mean", "DM": round(dm, 4), "p_value": round(pv, 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No forecast files found. Run proposed_seed*.py, baselines.py (and ablation.py) first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "DM_results.csv"), index=False)
    avg = out[out["Seed"].astype(str) == "mean"]
    wide = avg.pivot_table(index=["Loss", "Compared_with"], columns="Stock", values=["DM", "p_value"])
    print(avg.to_string(index=False))
    wide.to_csv(os.path.join(OUT_DIR, "DM_summary_seed_averaged.csv"))
    print(f"\nSaved to {OUT_DIR}/: DM_results.csv, DM_summary_seed_averaged.csv")
