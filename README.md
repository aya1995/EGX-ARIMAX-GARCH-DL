# ARIMAX-GARCH-Deep Learning Hybrid for Stock Price Forecasting (Egyptian Exchange)

Code accompanying the manuscript:
**"A Novel Hybrid Learning Approach Integrating ARIMAX-GARCH-Deep Learning for Stock Price Forecasting: An Empirical Study on the Egyptian Stock Market"**

Stocks studied: MASR, CCAP, PHDC, HELI.

## Method in brief
1. ARIMAX models the conditional mean of the next-day log return.
2. GARCH models the conditional volatility of the ARIMAX residuals (test-period volatility is forecast recursively from training data only).
3. A deep-learning model (LSTM, GRU or LSTM-GRU) predicts the ARIMAX residual from the lagged residual, six technical features and the GARCH volatility.
4. Final forecast = ARIMAX forecast + predicted residual (clipped to +/-30 %).

## Repository layout
```
src/
  common.py               shared code: settings, (STOCK_CONFIG), data/features/split, ARIMAX, GARCH,
                          DL inputs and models, accuracy metrics, long-only strategy with transaction costs
  grid_search.py          ARIMAX / GARCH / DL hyper-parameter search (training split only)  -> Table 3.5
  proposed_seed42.py      proposed model (AG-LSTM, AG-GRU, AG-LSTM-GRU), seed 42
  proposed_seed77.py      proposed model, seed 77
  proposed_seed123.py     proposed model, seed 123
  proposed_mean_std.py    mean +/- std of the proposed model over the three seed
  baselines.py            Buy & Hold, Naive, RW+Drift, ARIMAX, Random Forest, XGBoost
  ablation.py             ARIMAX, ARIMAX-GARCH, ARIMAX+DL (+ full model) over the three seeds
  friedman_finner.py      Friedman test (+ Iman-Davenport) and Finner post-hoc
  dm_test.py              Diebold-Mariano test, reference model AG-LSTM-GRU
  tscv_crisis.py          5-fold expanding-window time-series CV + crisis-period analysis (AG-LSTM-GRU only)
  shap_analysis.py        SHAP importance of the DL correction of AG-LSTM-GRU, PHDC, seed 42 (standalone script, needs `shap`)
data/                     <STOCK>_features_usd_egp.csv (see "Data")
results/                  all output files
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Data
Each stock is read from `data/<STOCK>_features_usd_egp.csv` (or the working directory) with columns
`Date, Open, High, Low, Close, Volume, USD_EGP`.
Data source: all series (daily Open, High, Low, Close, Volume of the Egyptian Exchange tickers, and the USD/EGP exchange rate in the `USD_EGP` column) were downloaded from the **Yahoo Finance API**.
Data period: **January 2010 - March 2026** (daily); downloaded in late March 2026. The first rows are lost to the warm-up of the technical indicators and the lagged features.
Yahoo Finance data are subject to Yahoo's terms of use, so the raw CSV files are **not included** in this repository: download the same series for each ticker and arrange them in the format above. Results can differ slightly if the data are downloaded at a different date (adjusted prices may be revised).
Folders can be changed with the environment variables `EGX_DATA_DIR` and `EGX_OUT_DIR`.

## Running (run from the repository root)
In Jupyter / Colab: upload the folder, `%cd egx-arimax-garch-dl`, `!pip install -r requirements.txt`, then run each script with `!python src/<script>.py ALL` (do not paste the script into a cell: `common.py` must be importable, i.e. run the file, or use `%run src/<script>.py ALL`).
Every stock script takes a stock name (`MASR`, `CCAP`, `PHDC`, `HELI`) or `ALL` (default).
```bash
python src/grid_search.py ALL            # 0. optional: reproduces (values are stored in common.py)

python src/proposed_seed42.py  ALL       # 2. proposed model, one file per seed
python src/proposed_seed77.py  ALL
python src/proposed_seed123.py ALL
python src/proposed_mean_std.py          #    mean +/- std over the three seeds

python src/baselines.py ALL              # 3. baselines
python src/ablation.py  ALL              # 4. ablation (run after step 2: it appends the full-model rows)

python src/friedman_finner.py            # 5. Friedman + Finner  (needs steps 2-3, all four stocks)
python src/dm_test.py                    # 6. Diebold-Mariano     (needs steps 2-4)
python src/tscv_crisis.py ALL            # 7. time-series CV + crisis analysis (independent of steps 2-6)
python src/shap_analysis.py              # 8. SHAP of AG-LSTM-GRU (PHDC; edit FILE_NAME for another stock)
```

## Evaluation protocol
- Chronological 80/20 train/test split, no shuffling. Hyper-parameters are chosen on the training split only.
- **Seeds 42, 77, 123 are fixed a priori.** Stochastic models (deep-learning models, Random Forest) are run with all
  three seeds and reported as mean +/- standard deviation (sample std, ddof = 1).
- Deterministic models (Buy & Hold, Naive, RW+Drift, ARIMAX, ARIMAX-GARCH) are run once and shown as a single value.
  XGBoost (`max_depth=3`, default sub-sampling = 1) gives identical results for every seed, so it is also a single value.
- Naive: previous observed return; RW+Drift: Naive + mean training return. Their trading metrics can coincide
  (e.g. when the training drift is too small to change any sign).
- Trading strategy: long-only, position = 1 when the forecast is positive. Total return = (exp(sum r) - 1) x 100,
  Sharpe = mean/std x sqrt(252). Costs are charged on every change of position (0 % = gross, 0.15 % per position change = net).
  Buy & Hold is always long (one entry).
- Directional accuracy (DA) is computed on days with a non-zero actual return.
- **ARIMAX-GARCH (ablation, forecasting model)**: GARCH(p,q) is fitted on the ARIMAX training residuals, the ARIMAX mean equation
  (same order and exogenous variables) is re-estimated on data divided by the GARCH volatility (weights 1/sigma^2, FGLS-style), and the
  test forecast is rescaled by the multi-step GARCH volatility forecast built from training data only. Its accuracy metrics differ from ARIMAX.
  This is a pragmatic two-step approximation of an ARIMAX model with GARCH errors, not a joint maximum-likelihood estimate.
- The GARCH contribution inside the proposed model (GARCH volatility as an input of the DL network) is isolated by *full model vs ARIMAX+DL* (same DL without the GARCH input).
- Statistical tests: Friedman test on the model means over seeds with N = 4 stocks (low power; several metrics are analysed,
  so read the results as descriptive evidence); Diebold-Mariano test (squared and absolute error, h = 1, HLN correction)
  of AG-LSTM-GRU against every other model, per seed and on the seed-averaged loss differential.

## Outputs (in `results/`)
| Script | Main files |
|---|---|
| `proposed_seed*.py` | `Results_proposed_<S>_seed<N>.csv`, `Predictions_<S>_proposed_seed<N>.csv` |
| `ablation_no_usd.py` | `NoUSD_per_seed_<S>.csv`, `NoUSD_mean_std_<S>.csv`, `NoUSD_mean_numeric_<S>.csv`, `NoUSD_mean_all_stocks.csv` |
| `proposed_ag_lstm_gru.py` | `AG_LSTM_GRU_per_seed_<S>.csv`, `AG_LSTM_GRU_mean_std_<S>.csv`, `AG_LSTM_GRU_mean_numeric_<S>.csv`, `AG_LSTM_GRU_mean_all_stocks.csv` |
| `proposed_mean_std.py` | `Proposed_mean_std_<S>.csv` (text "mean ± std"), `Proposed_mean_numeric_<S>.csv`, `Proposed_mean_of_3_models.csv` |
| `baselines.py` | `Baselines_per_seed_<S>.csv`, `Baselines_mean_std_<S>.csv`, `Baselines_mean_numeric_<S>.csv`, `Predictions_<S>_baselines_seed<N>.csv` |
| `ablation.py` | `Ablation_per_seed_<S>.csv`, `Ablation_mean_std_<S>.csv`, `Ablation_mean_of_DL_models_<S>.csv` |
| `friedman_finner.py` | `Friedman_avg_ranks.csv`, `Friedman_summary.csv`, `Finner_posthoc.csv` |
| `dm_test.py` | `DM_results.csv`, `DM_summary_seed_averaged.csv` |
| `grid_search.py` | `GridSearch_*_<S>.csv`, `GridSearch_Best_<S>.json` |
|
Rounding used in the paper tables: MSE 6 decimals; RMSE, MAE, DA and Sharpe 4; returns 2; number of trades integer.

## License
This code is released under the MIT License (see the `LICENSE` file).

## Citation
The accompanying manuscript is under review (journal: *Array*, Elsevier); the full citation and DOI will be added after acceptance.

## Time-series cross-validation and crisis analysis (`tscv_crisis.py`)
- Applied only to the proposed AG-LSTM-GRU, with exactly the same features, Table 3.5 parameters, GARCH forecast, clipping,
  long-only strategy and costs as the main experiments (all code is taken from `common.py`), and a single seed (42; `TSCV_SEEDS` at the top of the script, because 5 folds x 4 stocks x several seeds is very heavy).
- Five expanding-window folds (training always starts at the first observation). Test windows are half-open, so no date
  belongs to two folds: 2019-09-01..2020-12-31, 2020-12-31..2022-04-01, 2022-04-01..2023-08-01, 2023-08-01..2024-11-28,
  2024-11-28..2026-03-31. The first 20 test days of every fold are used as the input window of the DL model (as in the main split).
- ARIMAX and GARCH are re-estimated on each training window; if the Table 3.5 ARIMAX order does not converge on a window the script
  prints a message and falls back to ARIMAX(1,0,0) for that fold.
- Per fold: one run (with several seeds: mean, and std over seeds in `TSCV_Folds_<S>.csv`); the last row of `TSCV_paper_<S>.csv` is the mean +/- SD across the five folds.
- Crisis windows (COVID-19 2020-03-01..2020-12-31; Russia-Ukraine & inflation 2022-02-24..2023-08-01; Egyptian FX crisis 2023-08-01..2024-03-31;
  Gaza/Red Sea 2023-10-07..2024-11-28) use only the predictions dated inside the window. The last two windows overlap, so they are not independent.
  Each row also gives the Buy & Hold return of the same window. Windows with fewer than 10 predictions are skipped.
- Days with zero volume are not removed (same as the main experiments); set `FILTER_ZERO_VOLUME = True` only if the main data were filtered the same way.

## SHAP analysis (`shap_analysis.py`)
- Standalone script (it does not import `common.py`), used to produce the SHAP figures of the paper: PHDC, seed 42, KernelExplainer
  (100 background sequences, 200 explained test sequences, 100 coalitions per sample).
- The explained output is the DL correction (predicted ARIMAX residual in return units), not the full forecast: the ARIMAX part does not depend on the DL inputs.
  The importance of a variable is the mean |SHAP| averaged over its 20 lags.
- SHAP describes the trained model and is not a causal effect.
- Model settings come from a small Table 3.5 dictionary at the top of the script (ARIMAX order, GARCH(p,q), LSTM-GRU units, dropout, batch size of the chosen stock only; the stock is taken from `FILE_NAME`).
  Like all other scripts, it keeps zero-volume days.
