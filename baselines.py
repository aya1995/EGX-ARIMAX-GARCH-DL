# -*- coding: utf-8 -*-
"""
Baseline models, all in ONE script and ONE definition of data / split / metrics / strategy:
  Buy & Hold, Naive, RW+Drift, ARIMAX            (deterministic - single value)
  Random Forest, XGBoost                         (seeds 42, 77, 123 -> mean ± std)

Naive    : y_hat(t+1) = Log_Return(t-1)   (previous observed return; first test forecast = last training return)
RW+Drift : y_hat(t+1) = Naive + c         (c = mean of the TRAINING Log_Return only)
ARIMAX   : order from Table 3.5, exogenous variables = Open_return, HL_range_pct, Volume_change, USD_change
RF       : RandomForestRegressor(n_estimators=20),  XGBoost: XGBRegressor(max_depth=3)  on the 10 flat features
Strategy : long-only, position = 1 if forecast > 0 ; Buy & Hold = always long (one entry, one cost).

Usage:   python src/baselines.py MASR      (or CCAP / PHDC / HELI / ALL)
Output (per stock S):
  results/Baselines_per_seed_<S>.csv       every model x seed
  results/Baselines_mean_std_<S>.csv       paper table, text "mean ± std" (deterministic models: single value)
  results/Baselines_mean_numeric_<S>.csv   numeric means (input of friedman_finner.py)
  results/Predictions_<S>_baselines_seed<N>.csv   forecasts (input of dm_test.py)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from common import (SEEDS, SL, CLIP, METRIC_COLS, Prepared, stocks_from_argv, out_path,
                    fit_arimax, evaluate, strategy_with_position, accuracy_metrics,
                    COST_BASE, mean_std_tables, set_seed)

DETERMINISTIC = ["Buy & Hold", "Naive", "RW+Drift", "ARIMAX"]      # single run; XGBoost is shown as a single value only if its std over the seeds is exactly 0


def naive_forecasts(P):
    """Naive and RW+Drift forecasts aligned with P.actual_return (test rows SL:)."""
    full = np.zeros(len(P.test_df))
    full[0] = P.train_df["Log_Return"].iloc[-1]
    for i in range(1, len(P.test_df)):
        full[i] = P.test_df["Log_Return"].iloc[i - 1]
    naive = full[SL:]
    drift = P.train_df["Log_Return"].mean()          # training data only
    return naive, naive + drift, drift


def buy_and_hold_row(actual):
    pos = np.ones(len(actual))
    mse = rmse = mae = da = np.nan                    # no forecast -> no accuracy metrics
    rg, sg, nt = strategy_with_position(actual, pos, 0.0)
    rb, sb, _  = strategy_with_position(actual, pos, COST_BASE)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "DA": da,
            "Ret_gross_%": rg, "Sharpe_gross": sg, "N_Trades": nt,
            "Ret_net_0.15_%": rb, "Sharpe_net_0.15": sb}


def run_stock(s):
    from xgboost import XGBRegressor
    P = Prepared(s)
    actual = P.actual_return
    naive, rw, drift = naive_forecasts(P)
    print(f"  training drift = {drift:.8f}")
    ar = fit_arimax(P)
    arimax = np.clip(ar["arimax_return"], -CLIP, CLIP)
    assert len(naive) == len(rw) == len(arimax) == len(actual)

    rows = [{"Model": "Buy & Hold", "Seed": np.nan, **buy_and_hold_row(actual)}]
    det = {"Naive": naive, "RW+Drift": rw, "ARIMAX": arimax}
    for name, pr in det.items():
        rows.append({"Model": name, "Seed": np.nan, **evaluate(actual, pr)})

    for sd in SEEDS:
        set_seed(sd)
        rf  = RandomForestRegressor(n_estimators=20, random_state=sd, n_jobs=-1).fit(P.train_X, P.train_target)
        xgb = XGBRegressor(n_estimators=70,max_depth=3, random_state=sd).fit(P.train_X, P.train_target)
        pr = {"RF":      np.clip(np.nan_to_num(rf.predict(P.test_X)[SL:]),  -CLIP, CLIP),
              "XGBoost": np.clip(np.nan_to_num(xgb.predict(P.test_X)[SL:]), -CLIP, CLIP)}
        for name, p_ in pr.items():
            rows.append({"Model": name, "Seed": sd, **evaluate(actual, p_)})
        pd.DataFrame({"Date": P.pred_dates, "Actual": actual, **det, **pr}).to_csv(
            out_path(f"Predictions_{s}_baselines_seed{sd}.csv"), index=False)
        print(f"  seed {sd}: RF + XGBoost done")

    per = pd.DataFrame(rows).round(6)
    per.insert(0, "Stock", s)
    per.to_csv(out_path(f"Baselines_per_seed_{s}.csv"), index=False)

    txt, num, mean_only = mean_std_tables(per, deterministic=DETERMINISTIC)
    order = ["Buy & Hold", "Naive", "RW+Drift", "ARIMAX", "RF", "XGBoost"]
    txt = txt.set_index("Model").loc[order].reset_index()
    mean_only = mean_only.set_index("Model").loc[order].reset_index()
    txt.to_csv(out_path(f"Baselines_mean_std_{s}.csv"), index=False, encoding="utf-8-sig")
    mean_only.to_csv(out_path(f"Baselines_mean_numeric_{s}.csv"), index=False)
    print(f"\n{s}: baselines (mean ± std over seeds {SEEDS}; deterministic models single value)")
    print(txt.to_string(index=False))
    xg = per[per["Model"] == "XGBoost"]
    print("XGBoost identical across seeds (deterministic)?",
          bool(xg[["MSE", "Ret_gross_%", "N_Trades"]].nunique().max() == 1))


if __name__ == "__main__":
    for s in stocks_from_argv():
        print("=" * 70); print(f"BASELINES - {s}"); print("=" * 70)
        run_stock(s)
