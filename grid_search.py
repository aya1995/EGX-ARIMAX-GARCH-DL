# -*- coding: utf-8 -*-
"""
Grid search for ARIMAX-GARCH-DL  (run BEFORE the other scripts; its output is Table 3.5)
- Uses the TRAINING split only (first 80 %). The test set is never used here.
- ARIMAX order : lowest AIC
- GARCH (p,q)  : lowest AIC on the ARIMAX training residuals
- DL params    : lowest validation loss (last 10 % of the training sequences, chronological) per
                 architecture, with the same inputs as the proposed model (lagged residual + market
                 features + GARCH volatility).
Data, features and split come from common.py (identical to every other script).

Usage:   python src/grid_search.py MASR      (or CCAP / PHDC / HELI / ALL)
Output:  results/GridSearch_ARIMAX_<S>.csv, GridSearch_GARCH_<S>.csv, GridSearch_DL_<S>.csv,
         results/GridSearch_Best_<S>.json  + a STOCK_CONFIG snippet printed at the end
         (paste it into STOCK_CONFIG of common.py if the selected values differ from Table 3.5).
"""
import itertools, json
import numpy as np, pandas as pd
from common import (Prepared, stocks_from_argv, out_path, set_seed, SL, DL_FEATURES,
                    EPOCHS, PATIENCE)

SEED = 42
GS_EPOCHS, GS_PATIENCE = EPOCHS, PATIENCE

# ---- Search spaces (as in the paper table) ----
ARIMAX_SPACE = dict(p=[1, 2, 3], d=[0], q=[0, 1, 2])
GARCH_SPACE  = dict(p=[1, 2], q=[1, 2])
# LSTM-GRU: both layers use the same number of units, e.g. (32, 32)
DL_SPACE = {
    "LSTM":     dict(units=[32, 64],             dropout=[0.1, 0.2], batch=[32, 64]),
    "GRU":      dict(units=[32, 64],             dropout=[0.1, 0.2], batch=[32, 64]),
    "LSTM-GRU": dict(units=[(32, 32), (64, 64)], dropout=[0.1, 0.2], batch=[32, 64]),
}


def run_stock(S):
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.preprocessing import StandardScaler
    from arch import arch_model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    set_seed(SEED)
    P = Prepared(S)
    y_tr, exog, market = P.train_target, P.train_exog, P.train_market     # training part only

    # ---- 1. ARIMAX (AIC) ----
    print("\n--- ARIMAX grid ---")
    gs_ar, best = [], None
    for p, d, q in itertools.product(ARIMAX_SPACE["p"], ARIMAX_SPACE["d"], ARIMAX_SPACE["q"]):
        try:
            f = ARIMA(y_tr, exog=exog, order=(p, d, q)).fit()
            gs_ar.append({"order": f"({p},{d},{q})", "AIC": round(f.aic, 2), "BIC": round(f.bic, 2)})
            if best is None or f.aic < best[0]:
                best = (f.aic, (p, d, q), f)
            print(f"  ARIMAX({p},{d},{q}) AIC={f.aic:.2f}")
        except Exception as e:
            print(f"  ARIMAX({p},{d},{q}) failed: {e}")
    best_order, af = best[1], best[2]
    print(f"Best ARIMAX: {best_order}")
    resid = y_tr - np.asarray(af.predict(start=0, end=len(y_tr) - 1, exog=exog)).flatten()

    # ---- 2. GARCH (AIC) ----
    print("\n--- GARCH grid ---")
    gs_g, bestg = [], None
    for gp, gq in itertools.product(GARCH_SPACE["p"], GARCH_SPACE["q"]):
        try:
            g = arch_model(resid, mean="Zero", vol="GARCH", p=gp, q=gq, rescale=False).fit(disp="off")
            gs_g.append({"order": f"({gp},{gq})", "AIC": round(g.aic, 2), "BIC": round(g.bic, 2)})
            if bestg is None or g.aic < bestg[0]:
                bestg = (g.aic, (gp, gq), g)
            print(f"  GARCH({gp},{gq}) AIC={g.aic:.2f}")
        except Exception as e:
            print(f"  GARCH({gp},{gq}) failed: {e}")
    best_gpq, gf = bestg[1], bestg[2]
    print(f"Best GARCH(p,q): {best_gpq}")
    gvol = np.asarray(gf.conditional_volatility).flatten()

    # ---- 3. DL inputs (same construction as the proposed model, training part only) ----
    r_sc = StandardScaler().fit_transform(resid.reshape(-1, 1))
    m_sc = StandardScaler().fit_transform(market)
    g_sc = StandardScaler().fit_transform(gvol.reshape(-1, 1))
    r_lag = np.roll(r_sc, 1, axis=0); r_lag[0] = 0.0
    feat = np.concatenate([r_lag, m_sc, g_sc], axis=1)
    X = np.array([feat[i - SL:i] for i in range(SL, len(feat))])
    y = np.array([r_sc.flatten()[i] for i in range(SL, len(feat))])

    def build(mt, units, drop):
        shape = (X.shape[1], X.shape[2])
        if mt == "LSTM":
            return Sequential([LSTM(units, input_shape=shape), Dropout(drop), Dense(1)])
        if mt == "GRU":
            return Sequential([GRU(units, input_shape=shape), Dropout(drop), Dense(1)])
        u1, u2 = units
        return Sequential([LSTM(u1, return_sequences=True, input_shape=shape),
                           GRU(u2), Dropout(drop), Dense(1)])

    print("\n--- DL grid (validation = last 10 % of training sequences) ---")
    gs_dl, best_dl = [], {}
    for mt, sp in DL_SPACE.items():
        for units, drop, batch in itertools.product(sp["units"], sp["dropout"], sp["batch"]):
            tf.random.set_seed(SEED)
            m = build(mt, units, drop); m.compile(optimizer="adam", loss="mse")
            h = m.fit(X, y, validation_split=0.1, epochs=GS_EPOCHS, batch_size=batch, verbose=0,
                      shuffle=False,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=GS_PATIENCE,
                                               restore_best_weights=True)])
            vl = float(min(h.history["val_loss"]))
            gs_dl.append({"arch": mt, "units": str(units), "dropout": drop, "batch": batch,
                          "val_loss": round(vl, 6)})
            print(f"  {mt}(u={units},dr={drop},b={batch}) val={vl:.6f}")
            if mt not in best_dl or vl < best_dl[mt]["val_loss"]:
                best_dl[mt] = {"units": units, "dropout": drop, "batch": batch, "val_loss": vl}

    # ---- Save ----
    pd.DataFrame(gs_ar).sort_values("AIC").to_csv(out_path(f"GridSearch_ARIMAX_{S}.csv"), index=False)
    pd.DataFrame(gs_g).sort_values("AIC").to_csv(out_path(f"GridSearch_GARCH_{S}.csv"), index=False)
    pd.DataFrame(gs_dl).sort_values(["arch", "val_loss"]).to_csv(out_path(f"GridSearch_DL_{S}.csv"), index=False)
    with open(out_path(f"GridSearch_Best_{S}.json"), "w") as fh:
        json.dump({"stock": S, "arimax_order": best_order, "garch_pq": best_gpq, **best_dl}, fh, indent=2)

    print("\n# ---- paste into STOCK_CONFIG of common.py (only if it differs from Table 3.5) ----")
    print(f'"{S}": {{"arimax_order": {best_order}, "garch_pq": {best_gpq},')
    for mt in ["LSTM", "GRU", "LSTM-GRU"]:
        b = best_dl[mt]
        print(f'    "{mt}": {{"units": {b["units"]}, "dropout": {b["dropout"]}, "batch": {b["batch"]}}},')
    print("},")


if __name__ == "__main__":
    for s in stocks_from_argv():
        print("=" * 70); print(f"GRID SEARCH - {s}"); print("=" * 70)
        run_stock(s)
