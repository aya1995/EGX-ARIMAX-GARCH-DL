# -*- coding: utf-8 -*-
"""
COMPONENT ABLATION based on ARIMAX (seeds 42, 77, 123; mean ± std)

  ARIMAX                   ARIMAX point forecast.                                   (deterministic)
  ARIMAX-GARCH             GARCH changes the FORECAST (volatility-weighted ARIMAX, FGLS-style):
                           GARCH(p,q) is fitted on the ARIMAX training residuals; the ARIMAX mean equation
                           (same order, same exogenous variables) is then re-estimated on data divided by the
                           GARCH volatility sigma_t (weights 1/sigma_t^2), and the test forecast is rescaled by the
                           multi-step GARCH volatility forecast (training data only, no leakage). Accuracy metrics
                           differ from ARIMAX.                                              (deterministic)
  ARIMAX+DL-<type>         Same DL models / parameters as the proposed model but WITHOUT the GARCH
                           volatility input (lagged residual + market features only).      (3 seeds)
  ARIMAX+GARCH+DL-<type>   Full proposed model - read from results/Results_proposed_<S>_seed*.csv
                           (run proposed_seed*.py first; skipped with a message if missing).

Comparison that isolates the GARCH contribution inside the proposed model:
  ARIMAX+GARCH+DL (full)  vs  ARIMAX+DL (same DL, no GARCH input).

Usage:   python src/ablation.py MASR      (or CCAP / PHDC / HELI / ALL)
Output (per stock S):
  results/Ablation_per_seed_<S>.csv, Ablation_mean_std_<S>.csv (text), Ablation_mean_numeric_<S>.csv,
  results/Ablation_mean_of_DL_models_<S>.csv   (ARIMAX+DL and full model averaged over LSTM/GRU/LSTM-GRU)
  results/Predictions_<S>_ablation_seed<N>.csv (input of dm_test.py)
"""
import numpy as np
import pandas as pd
from common import (SEEDS, SL, CLIP, METRIC_COLS, DL_TYPES, AG_NAMES, NOG_NAMES, Prepared,
                    stocks_from_argv, out_path, fit_arimax, fit_garch, make_dl_data,
                    build_and_train_dl, dl_forecast, evaluate, mean_std_tables, round_mean_only,
                    set_seed)

DETERMINISTIC = ["ARIMAX", "ARIMAX-GARCH"]


def arimax_garch_forecast(P, ar, gv):
    """Volatility-weighted ARIMAX (FGLS-style): ARIMAX re-estimated on y/sigma_t with regressors X/sigma_t
    and 1/sigma_t (constant), same order as Table 3.5. Test forecast = model forecast x forecast sigma."""
    from statsmodels.tsa.arima.model import ARIMA
    s_tr = np.maximum(gv["train_vol"], 1e-8)
    s_te = np.maximum(gv["test_vol"], 1e-8)
    X_tr = np.column_stack([P.train_exog / s_tr[:, None], 1.0 / s_tr])
    X_te = np.column_stack([P.test_exog / s_te[:, None], 1.0 / s_te])
    order = P.cfg["arimax_order"]
    try:
        fit = ARIMA(P.train_target / s_tr, exog=X_tr, order=order, trend="n").fit()
    except Exception as e:
        print(f"    ARIMAX-GARCH: order {order} failed ({e}); using (1,0,0)")
        fit = ARIMA(P.train_target / s_tr, exog=X_tr, order=(1, 0, 0), trend="n").fit()
    fc = np.asarray(fit.forecast(steps=len(P.test_target), exog=X_te)).flatten() * s_te
    return fc[SL:]


def run_stock(s):
    P = Prepared(s)
    ar = fit_arimax(P)
    arx = np.clip(ar["arimax_return"], -CLIP, CLIP)

    # ---- ARIMAX-GARCH: GARCH changes the forecast ----
    gv_fc = fit_garch(P, ar)
    ag_fc = np.clip(np.nan_to_num(arimax_garch_forecast(P, ar, gv_fc)), -CLIP, CLIP)

    rows = [{"Model": "ARIMAX", "Seed": np.nan, **evaluate(P.actual_return, arx)},
            {"Model": "ARIMAX-GARCH", "Seed": np.nan, **evaluate(P.actual_return, ag_fc)}]

    # ---- ARIMAX+DL (no GARCH input) ----
    data = make_dl_data(P, ar, gv=None)
    for sd in SEEDS:
        set_seed(sd)
        preds = {"ARIMAX": arx, "ARIMAX-GARCH": ag_fc}
        for mt in DL_TYPES:
            m = build_and_train_dl(mt, P.cfg, data, sd)
            pr = dl_forecast(m, data, ar["arimax_return"])
            preds[NOG_NAMES[mt]] = pr
            rows.append({"Model": NOG_NAMES[mt], "Seed": sd, **evaluate(P.actual_return, pr)})
            print(f"  seed {sd}: {NOG_NAMES[mt]} done")
        pd.DataFrame({"Date": P.pred_dates, "Actual": P.actual_return, **preds}).to_csv(
            out_path(f"Predictions_{s}_ablation_seed{sd}.csv"), index=False)

    # ---- full model (from the proposed-model files) ----
    full_rows, have_full = [], True
    for sd in SEEDS:
        try:
            f = pd.read_csv(out_path(f"Results_proposed_{s}_seed{sd}.csv"))
        except FileNotFoundError:
            have_full = False; break
        for _, r in f.iterrows():
            full_rows.append({"Model": r["Model"].replace("AG-", "ARIMAX+GARCH+DL-"), "Seed": sd,
                              **{c: r[c] for c in METRIC_COLS}})
    if have_full:
        rows += full_rows
    else:
        print(f"[{s}] full-model rows skipped: run proposed_seed*.py first")

    per = pd.DataFrame(rows).round(6)
    per.insert(0, "Stock", s)
    per.to_csv(out_path(f"Ablation_per_seed_{s}.csv"), index=False)
    txt, num, mean_only = mean_std_tables(per, deterministic=DETERMINISTIC)
    txt.to_csv(out_path(f"Ablation_mean_std_{s}.csv"), index=False, encoding="utf-8-sig")
    mean_only.to_csv(out_path(f"Ablation_mean_numeric_{s}.csv"), index=False)

    # ---- one row per component: average over the three DL architectures ----
    def grp(m):
        if m.startswith("ARIMAX+GARCH+DL"): return "ARIMAX+GARCH+DL (mean of 3)"
        if m.startswith("ARIMAX+DL"):       return "ARIMAX+DL (mean of 3)"
        return m
    g = mean_only.copy(); g["Group"] = g["Model"].map(grp)
    g = g.groupby("Group", sort=False)[METRIC_COLS].mean().reset_index().rename(columns={"Group": "Model"})
    g.insert(0, "Stock", s)
    round_mean_only(g).to_csv(out_path(f"Ablation_mean_of_DL_models_{s}.csv"), index=False)

    print(f"\n{s}: ablation (mean ± std over seeds {SEEDS}; ARIMAX, ARIMAX-GARCH deterministic)")
    print(txt.to_string(index=False))


if __name__ == "__main__":
    for s in stocks_from_argv():
        print("=" * 70); print(f"ABLATION - {s}"); print("=" * 70)
        run_stock(s)
