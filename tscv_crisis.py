# -*- coding: utf-8 -*-
"""
Time-series cross-validation (5 expanding-window folds) + crisis-period analysis
for the proposed model AG-LSTM-GRU only (ARIMAX -> GARCH -> LSTM-GRU).

Same as the main experiments (everything comes from common.py):
  * same features (ARIMAX: 4 exogenous variables; DL: lagged residual + 6 market features + GARCH volatility)
  * same Table 3.5 parameters per stock (ARIMAX order, GARCH(p,q), LSTM-GRU units / dropout / batch size)
  * same multi-step GARCH volatility forecast (built from the training window only)
  * same forecast = ARIMAX forecast + predicted residual, clipped to +/-30 %
  * same long-only strategy: gross and net of 0.15 % per position change
  * ONE seed (42; TSCV_SEEDS at the top): fold values are single runs, the only spread reported is
    the SD ACROSS the five folds. If more seeds are listed, per-fold std over seeds is reported as well.

Folds (train always starts at the first observation; test windows are half-open [start, end),
the last one is closed, so no date is in two test windows):
  1: test 2019-09-01 .. 2020-12-31      4: test 2023-08-01 .. 2024-11-28
  2: test 2020-12-31 .. 2022-04-01      5: test 2024-11-28 .. 2026-03-31
  3: test 2022-04-01 .. 2023-08-01
Crisis analysis: ONLY predictions whose dates fall inside each crisis window are used (windows may overlap).

Usage:   python src/tscv_crisis.py MASR        (or CCAP / PHDC / HELI / ALL)
Output (results/):
  TSCV_Folds_<S>.csv              per fold, mean over seeds (numeric) + std over seeds
  TSCV_per_seed_<S>.csv           per fold and seed
  TSCV_paper_<S>.csv              paper format: Fold 1-5 and "Mean +/- SD" across folds
  TSCV_Predictions_<S>.csv        date, fold, actual, forecast of each seed
  Crisis_Analysis_<S>.csv         per crisis window: mean +/- std over seeds, with Buy & Hold of the same window
"""
import numpy as np
import pandas as pd
from common import (SL, COST_BASE, DECIMALS, Prepared, load_features, fit_arimax, fit_garch,
                    make_dl_data, build_and_train_dl, dl_forecast, evaluate, set_seed,
                    stocks_from_argv, out_path, data_path, AG_NAMES)

MODEL = "LSTM-GRU"
TSCV_SEEDS = [42]            # seed 42 only: 5 folds x 3 seeds x 4 stocks is too heavy; add 77, 123 here if you have the time
FILTER_ZERO_VOLUME = False   # the main experiments do NOT drop zero-volume days; keep False for consistency

# (train_start, test_start, test_end, label)   train = [train_start, test_start), test = [test_start, test_end)
FOLDS = [("2010-01-01", "2019-09-01", "2020-12-31", "COVID-19"),
         ("2010-01-01", "2020-12-31", "2022-04-01", "COVID aftermath + Russia-Ukraine"),
         ("2010-01-01", "2022-04-01", "2023-08-01", "Russia-Ukraine + inflation"),
         ("2010-01-01", "2023-08-01", "2024-11-28", "Egyptian financial/FX crisis"),
         ("2010-01-01", "2024-11-28", "2026-03-31", "Post-floatation + geopolitical tensions")]

CRISES = {"COVID-19 Pandemic":               ("2020-03-01", "2020-12-31"),
          "Russia-Ukraine War & Inflation":  ("2022-02-24", "2023-08-01"),
          "Egyptian Financial/FX Crisis":    ("2023-08-01", "2024-03-31"),
          "Gaza War & Red Sea Tensions":     ("2023-10-07", "2024-11-28")}   # overlaps the FX-crisis window
MIN_OBS = 10

EXTRA = ["MAPE", "SMAPE"]
COLS = ["RMSE", "MAE", "MAPE", "SMAPE", "DA", "Ret_gross_%", "Sharpe_gross", "N_Trades",
        "Ret_net_0.15_%", "Sharpe_net_0.15"]
DEC = {**DECIMALS, "MAPE": 2, "SMAPE": 2}


def mape(a, p):
    m = np.abs(a) > 1e-10
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else np.nan


def smape(a, p):
    d = (np.abs(a) + np.abs(p)) / 2
    m = d > 1e-10
    return float(np.mean(np.abs(a[m] - p[m]) / d[m]) * 100) if m.any() else np.nan


def metrics(actual, pred):
    r = evaluate(actual, pred)
    r["MAPE"], r["SMAPE"] = mape(actual, pred), smape(actual, pred)
    return {c: r[c] for c in COLS}


def pm(mean, std, col):
    return f"{mean:.{DEC[col]}f} ± {std:.{DEC[col]}f}" if not pd.isna(std) else f"{mean:.{DEC[col]}f}"


def fold_positions(df):
    out = []
    last = len(FOLDS) - 1
    for k, (a, t0, t1, lab) in enumerate(FOLDS):
        idx = df.index
        tr = np.where((idx >= a) & (idx < t0))[0]
        te = np.where((idx >= t0) & ((idx <= t1) if k == last else (idx < t1)))[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        out.append((k + 1, lab, tr[-1] + 1, te[0], te[-1] + 1))
    return out


def run_stock(stock):
    df = load_features(stock)
    if FILTER_ZERO_VOLUME:
        raw = pd.read_csv(data_path(stock), usecols=["Date", "Volume"], parse_dates=["Date"])
        df = df[df.index.isin(raw.loc[raw["Volume"] > 0, "Date"])]
    print(f"[{stock}] {df.index[0].date()} -> {df.index[-1].date()}  rows={len(df)}")
    rows, preds = [], []
    for fold, label, n_train, t0, t1 in fold_positions(df):
        P = Prepared(stock, df=df, split=(n_train, t0, t1), verbose=False)
        print(f"  Fold {fold}: train {df.index[0].date()} -> {df.index[n_train-1].date()} ({n_train}) | "
              f"test {df.index[t0].date()} -> {df.index[t1-1].date()} ({t1-t0}) | {label}")
        if len(P.train_df) <= SL or len(P.test_df) <= SL:
            print("    skipped: not enough observations"); continue
        try:
            ar = fit_arimax(P)
        except Exception as e:                      # fallback, reported (never silent)
            print(f"    ARIMAX{P.cfg['arimax_order']} failed ({e}); using (1,0,0) for this fold")
            P.cfg["arimax_order"] = (1, 0, 0)
            ar = fit_arimax(P)
        gv = fit_garch(P, ar)
        data = make_dl_data(P, ar, gv)
        p_rows = {"Date": P.pred_dates, "Fold": fold, "Actual": P.actual_return}
        for seed in TSCV_SEEDS:
            set_seed(seed)
            model = build_and_train_dl(MODEL, P.cfg, data, seed)
            pred = dl_forecast(model, data, ar["arimax_return"])
            p_rows[f"Pred_seed{seed}"] = pred
            rows.append({"Stock": stock, "Fold": fold, "Seed": seed, "Test_start": str(P.pred_dates[0].date()),
                         "Test_end": str(P.pred_dates[-1].date()), "N_pred": len(pred),
                         **metrics(P.actual_return, pred)})
            print(f"    seed {seed}: RMSE={rows[-1]['RMSE']:.5f} DA={rows[-1]['DA']:.4f} "
                  f"Ret={rows[-1]['Ret_gross_%']:.2f}% Net={rows[-1]['Ret_net_0.15_%']:.2f}%")
        preds.append(pd.DataFrame(p_rows))
    per_seed = pd.DataFrame(rows)
    pdf = pd.concat(preds, ignore_index=True)
    pdf.to_csv(out_path(f"TSCV_Predictions_{stock}.csv"), index=False)
    per_seed.round(6).to_csv(out_path(f"TSCV_per_seed_{stock}.csv"), index=False)

    # ---- per fold: mean / std over seeds
    g = per_seed.groupby("Fold")[COLS]
    fm, fs = g.mean(), g.std(ddof=1)
    folds_out = fm.add_suffix("_mean").join(fs.add_suffix("_std")).reset_index()
    folds_out.insert(0, "Stock", stock)
    folds_out.round(6).to_csv(out_path(f"TSCV_Folds_{stock}.csv"), index=False)

    # ---- paper format: fold values = mean over seeds; last row = mean +/- SD across folds
    paper = []
    for f in fm.index:
        paper.append({"Stock": stock, "Fold": int(f), **{c: pm(fm.loc[f, c], fs.loc[f, c], c) for c in COLS}})
    cm, cs = fm.mean(), fm.std(ddof=1)
    paper.append({"Stock": stock, "Fold": "Mean ± SD", **{c: pm(cm[c], cs[c], c) for c in COLS}})
    pd.DataFrame(paper).to_csv(out_path(f"TSCV_paper_{stock}.csv"), index=False, encoding="utf-8-sig")

    # ---- crisis analysis (per seed on the pooled predictions inside each window)
    dates = pd.to_datetime(pdf["Date"])
    crows = []
    for name, (a, b) in CRISES.items():
        m = ((dates >= a) & (dates <= b)).values
        sub = pdf[m]
        if len(sub) < MIN_OBS:
            print(f"  {name}: not enough observations ({len(sub)})"); continue
        actual = sub["Actual"].values
        per = pd.DataFrame([metrics(actual, sub[f"Pred_seed{s}"].values) for s in TSCV_SEEDS])
        mu, sd = per.mean(), per.std(ddof=1)
        bh_g = (np.exp(actual.sum()) - 1) * 100
        bh_n = (np.exp(actual.sum() - COST_BASE) - 1) * 100
        crows.append({"Stock": stock, "Crisis": name, "Start": a, "End": b, "Observations": len(sub),
                      **{f"{c}_mean": mu[c] for c in COLS}, **{f"{c}_std": sd[c] for c in COLS},
                      "BuyHold_Ret_gross_%": bh_g, "BuyHold_Ret_net_0.15_%": bh_n,
                      **{f"{c} (mean ± std)": pm(mu[c], sd[c], c) for c in COLS}})
        print(f"  {name}: n={len(sub)} RMSE={mu['RMSE']:.5f} DA={mu['DA']:.4f} "
              f"Ret={mu['Ret_gross_%']:.2f}% (B&H {bh_g:.2f}%)")
    pd.DataFrame(crows).round(6).to_csv(out_path(f"Crisis_Analysis_{stock}.csv"), index=False, encoding="utf-8-sig")
    return per_seed


if __name__ == "__main__":
    for s in stocks_from_argv():
        print("=" * 70); print(f"TSCV + CRISIS - {s} - {AG_NAMES[MODEL]}"); print("=" * 70)
        run_stock(s)
