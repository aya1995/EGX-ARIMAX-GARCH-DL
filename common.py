# -*- coding: utf-8 -*-
"""
Shared code for the ARIMAX-GARCH-DL study (EGX: MASR, CCAP, PHDC, HELI).

Every script in this folder imports from here, so that all tables of the paper
come from ONE definition of: data/features, split, ARIMAX, GARCH, DL inputs,
DL models, accuracy metrics, long-only strategy and transaction costs.

Heavy libraries (statsmodels, arch, tensorflow) are imported lazily inside the
functions that need them, so the statistics scripts (Friedman, DM, mean/std) run
with pandas / numpy / scipy only.
"""
import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Settings (identical for every script)
# ----------------------------------------------------------------------------
STOCKS      = ["MASR", "CCAP", "PHDC", "HELI"]
SEEDS       = [42, 77, 123]        # fixed a priori; never changed after seeing results
TRAIN_RATIO = 0.80                 # chronological split, no shuffling
SL          = 20                   # sequence length of the DL models
EPOCHS      = 100
PATIENCE    = 15
CLIP        = 0.30                 # forecasts are clipped to +/-30 % daily return
COST_BASE   = 0.0015               # 0.15 % per unit of position change

ARIMAX_FEATURES = ["Open_return", "HL_range_pct", "Volume_change", "USD_change"]
DL_FEATURES     = ["Return_5D", "MACD_norm", "MACD_sig_norm",
                   "ATR_pct", "Stochastic", "EMA_10_norm"]

METRIC_COLS = ["MSE", "RMSE", "MAE", "DA", "Ret_gross_%", "Sharpe_gross", "N_Trades",
               "Ret_net_0.15_%", "Sharpe_net_0.15"]

# Decimals used when numbers are written into the paper tables
DECIMALS = {"MSE": 6, "RMSE": 4, "MAE": 4, "DA": 4,
            "Ret_gross_%": 2, "Sharpe_gross": 4, "N_Trades": 0,
            "Ret_net_0.15_%": 2, "Sharpe_net_0.15": 4}

# Table 3.5 (selected by grid_search.py on the training split only)
STOCK_CONFIG = {
    "MASR": {"arimax_order": (1, 0, 0), "garch_pq": (1, 1),
             "LSTM":     {"units": 32,       "dropout": 0.1, "batch": 64},
             "GRU":      {"units": 64,       "dropout": 0.2, "batch": 64},
             "LSTM-GRU": {"units": (32, 32), "dropout": 0.1, "batch": 64}},
    "CCAP": {"arimax_order": (3, 0, 0), "garch_pq": (1, 2),
             "LSTM":     {"units": 32,       "dropout": 0.1, "batch": 64},
             "GRU":      {"units": 32,       "dropout": 0.2, "batch": 64},
             "LSTM-GRU": {"units": (32, 32), "dropout": 0.1, "batch": 64}},
    "PHDC": {"arimax_order": (1, 0, 1), "garch_pq": (1, 2),
             "LSTM":     {"units": 64,       "dropout": 0.2, "batch": 32},
             "GRU":      {"units": 64,       "dropout": 0.2, "batch": 64},
             "LSTM-GRU": {"units": (32, 32), "dropout": 0.2, "batch": 32}},
    "HELI": {"arimax_order": (1, 0, 0), "garch_pq": (2, 2),
             "LSTM":     {"units": 32,       "dropout": 0.1, "batch": 64},
             "GRU":      {"units": 64,       "dropout": 0.1, "batch": 32},
             "LSTM-GRU": {"units": (32, 32), "dropout": 0.1, "batch": 64}},
}

DL_TYPES  = ["LSTM", "GRU", "LSTM-GRU"]
AG_NAMES  = {"LSTM": "AG-LSTM", "GRU": "AG-GRU", "LSTM-GRU": "AG-LSTM-GRU"}      # full model
NOG_NAMES = {"LSTM": "ARIMAX+DL-LSTM", "GRU": "ARIMAX+DL-GRU",
             "LSTM-GRU": "ARIMAX+DL-LSTM-GRU"}                                  # no GARCH input

DATA_DIR = os.environ.get("EGX_DATA_DIR", "data")
OUT_DIR  = os.environ.get("EGX_OUT_DIR", "results")


# ----------------------------------------------------------------------------
# Small helpers: command line, paths
# ----------------------------------------------------------------------------
def stocks_from_argv(default=None):
    """python script.py MASR  -> ['MASR'];  python script.py ALL (or nothing) -> all four."""
    default = default or STOCKS
    args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]   # ignore Jupyter/Colab flags
    if args and args[0].upper() != "ALL":
        s = args[0].upper()
        if s not in STOCK_CONFIG:
            raise SystemExit(f"Unknown stock '{s}'. Use one of {STOCKS} or ALL.")
        return [s]
    return list(default)


def data_path(stock):
    name = f"{stock}_features_usd_egp.csv"
    for d in (DATA_DIR, "."):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in '{DATA_DIR}/' or the working directory")


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def set_seed(seed):
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Data, features, split
# ----------------------------------------------------------------------------
def load_features(stock):
    df = pd.read_csv(data_path(stock))
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    df["Log_Return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["Target_Return"] = df["Log_Return"].shift(-1)                 # next-day log return
    df["Open_return"]   = np.log(df["Open"] / df["Open"].shift(1))
    df["USD_change"]    = df["USD_EGP"].pct_change() * 100
    df["Volume_change"] = df["Volume"].pct_change() * 100
    df["HL_range_pct"]  = (df["High"] - df["Low"]) / (df["Close"] + 1e-10) * 100
    ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_10_norm"]   = ema_10 / (df["Close"] + 1e-10)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]          = ema_12 - ema_26
    df["MACD_norm"]     = df["MACD"] / (df["Close"] + 1e-10)
    df["MACD_signal"]   = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_sig_norm"] = df["MACD_signal"] / (df["Close"] + 1e-10)
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["ATR_pct"]    = tr.rolling(14).mean() / (df["Close"] + 1e-10) * 100
    l14 = df["Low"].rolling(14).min()
    h14 = df["High"].rolling(14).max()
    df["Stochastic"] = 100 * ((df["Close"] - l14) / (h14 - l14 + 1e-10))
    df["Return_5D"]  = np.log(df["Close"] / df["Close"].shift(5))
    return df.replace([np.inf, -np.inf], np.nan).dropna().copy()


class Prepared:
    """Data of one stock after the split.
    Default: chronological 80/20 split. split=(train_end, test_start, test_end) (row positions)
    gives an arbitrary split, e.g. an expanding-window fold of the time-series cross-validation."""
    def __init__(self, stock, df=None, split=None, verbose=True):
        self.stock = stock
        self.cfg   = copy.deepcopy(STOCK_CONFIG[stock])
        self.df    = load_features(stock) if df is None else df
        if split is None:
            n = int(len(self.df) * TRAIN_RATIO)
            split = (n, n, len(self.df))
        n, t0, t1 = split
        self.n_train  = n
        self.train_df = self.df.iloc[:n].copy()
        self.test_df  = self.df.iloc[t0:t1].copy()
        self.train_target = self.train_df["Target_Return"].values.astype(float)
        self.test_target  = self.test_df["Target_Return"].values.astype(float)
        self.train_exog   = self.train_df[ARIMAX_FEATURES].values.astype(float)
        self.test_exog    = self.test_df[ARIMAX_FEATURES].values.astype(float)
        self.train_market = self.train_df[DL_FEATURES].values.astype(float)
        self.test_market  = self.test_df[DL_FEATURES].values.astype(float)
        self.actual_return = self.test_target[SL:]          # evaluation target
        self.pred_dates    = self.test_df.index[SL:]
        all_f = ARIMAX_FEATURES + DL_FEATURES                # flat features for RF / XGBoost
        self.train_X = self.train_df[all_f].values.astype(float)
        self.test_X  = self.test_df[all_f].values.astype(float)
        if verbose:
            print(f"[{stock}] rows={len(self.df)} | train={n} | test={len(self.test_df)}")


# ----------------------------------------------------------------------------
# ARIMAX and GARCH
# ----------------------------------------------------------------------------
def fit_arimax(P):
    """ARIMAX on the next-day return; returns forecasts and residuals."""
    from statsmodels.tsa.arima.model import ARIMA
    af = ARIMA(P.train_target, exog=P.train_exog, order=P.cfg["arimax_order"]).fit()
    train_ap = np.asarray(af.predict(start=0, end=len(P.train_target) - 1,
                                     exog=P.train_exog)).flatten()
    test_ap  = np.asarray(af.forecast(steps=len(P.test_target), exog=P.test_exog)).flatten()
    return {"model": af,
            "train_pred": train_ap, "test_pred": test_ap,
            "train_resid": P.train_target - train_ap,
            "test_resid":  P.test_target - test_ap,
            "arimax_return": test_ap[SL:]}


def fit_garch(P, ar):
    """GARCH(p,q) on the ARIMAX training residuals.
    Test volatility is a multi-step forecast built from TRAINING data only
    (future squared residuals are replaced by their expectation = forecast variance)."""
    from arch import arch_model
    G_P, G_Q = P.cfg["garch_pq"]
    train_resid = ar["train_resid"]
    gf = arch_model(train_resid, mean="Zero", vol="GARCH", p=G_P, q=G_Q, rescale=False).fit(disp="off")
    tr_gvol = np.asarray(gf.conditional_volatility).flatten()
    prm = gf.params
    om = float(prm.get("omega", 0))
    al = [float(prm.get(f"alpha[{i}]", 0)) for i in range(1, G_P + 1)]
    be = [float(prm.get(f"beta[{i}]", 0)) for i in range(1, G_Q + 1)]
    eps2 = list(train_resid[::-1][:G_P] ** 2)      # most recent first
    var  = list(tr_gvol[::-1][:G_Q] ** 2)
    te_gvol = []
    for _ in range(len(P.test_target)):
        nv = om + sum(al[i] * eps2[i] for i in range(G_P)) + sum(be[j] * var[j] for j in range(G_Q))
        nv = max(nv, 1e-12)
        te_gvol.append(np.sqrt(nv))
        eps2 = [nv] + eps2[:G_P - 1]
        var  = [nv] + var[:G_Q - 1]
    return {"model": gf, "train_vol": tr_gvol, "test_vol": np.array(te_gvol)}


# ----------------------------------------------------------------------------
# Deep-learning inputs and models
# ----------------------------------------------------------------------------
def make_sequences(f, t, sl):
    X, y = [], []
    for i in range(sl, len(f)):
        X.append(f[i - sl:i]); y.append(t[i])
    return np.array(X), np.array(y)


def make_dl_data(P, ar, gv=None):
    """Scaled DL inputs. gv=None -> inputs WITHOUT the GARCH volatility (ARIMAX+DL ablation).
    Inputs: lagged scaled ARIMAX residual + 6 market features (+ scaled GARCH volatility).
    The test-period residual input is never filled with test residuals (only the last
    training residual enters the first test step), so no test information leaks."""
    from sklearn.preprocessing import StandardScaler
    sc_r = StandardScaler()
    tr_r = sc_r.fit_transform(ar["train_resid"].reshape(-1, 1))
    te_r = sc_r.transform(ar["test_resid"].reshape(-1, 1))
    sc_m = StandardScaler()
    tr_m = sc_m.fit_transform(P.train_market)
    te_m = sc_m.transform(P.test_market)
    tr_rh = np.roll(tr_r, 1, axis=0); tr_rh[0] = 0.0
    te_rh = np.zeros_like(te_r);      te_rh[0] = tr_r[-1]
    tr_parts, te_parts = [tr_rh, tr_m], [te_rh, te_m]
    if gv is not None:
        sc_g = StandardScaler()
        tr_parts.append(sc_g.fit_transform(gv["train_vol"].reshape(-1, 1)))
        te_parts.append(sc_g.transform(gv["test_vol"].reshape(-1, 1)))
    tr_dl = np.concatenate(tr_parts, axis=1)
    te_dl = np.concatenate(te_parts, axis=1)
    Xtr, ytr = make_sequences(tr_dl, tr_r.flatten(), SL)
    Xte, _   = make_sequences(te_dl, te_r.flatten(), SL)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "sc_r": sc_r}


def build_and_train_dl(model_type, cfg, data, seed):
    """LSTM / GRU / LSTM-GRU with the Table 3.5 parameters; returns the fitted Keras model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(seed)                                   # same seed before every model
    c = cfg[model_type]
    shape = (data["Xtr"].shape[1], data["Xtr"].shape[2])
    if model_type == "LSTM":
        m = Sequential([LSTM(c["units"], input_shape=shape), Dropout(c["dropout"]), Dense(1)])
    elif model_type == "GRU":
        m = Sequential([GRU(c["units"], input_shape=shape), Dropout(c["dropout"]), Dense(1)])
    else:
        m = Sequential([LSTM(c["units"][0], return_sequences=True, input_shape=shape),
                        GRU(c["units"][1]), Dropout(c["dropout"]), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(data["Xtr"], data["ytr"], validation_split=0.1, epochs=EPOCHS,
          batch_size=c["batch"], verbose=0, shuffle=False,
          callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                   restore_best_weights=True)])
    return m


def dl_forecast(model, data, arimax_return):
    """Final forecast = ARIMAX forecast + predicted residual (back-transformed), clipped."""
    ps = model.predict(data["Xte"], verbose=0).flatten()
    pr = data["sc_r"].inverse_transform(ps.reshape(-1, 1)).flatten()
    return np.clip(np.nan_to_num(arimax_return + pr), -CLIP, CLIP)


# ----------------------------------------------------------------------------
# Metrics and long-only strategy with transaction costs
# ----------------------------------------------------------------------------
def accuracy_metrics(actual, pred):
    """MSE, RMSE, MAE and directional accuracy (DA).
    DA is computed only on days with a non-zero actual return (direction undefined otherwise)."""
    err  = actual - pred
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(err)))
    mask = actual != 0
    da   = float(np.mean(np.sign(actual[mask]) == np.sign(pred[mask]))) if mask.sum() else 0.0
    return mse, rmse, mae, da


def strategy_with_position(actual, position, cost_rate):
    """r_t = position_t * actual_t - cost * |position_t - position_{t-1}|  (start flat).
    Total return = (exp(sum r) - 1) * 100 ; Sharpe = mean/std * sqrt(252) (ddof = 0)."""
    prev  = np.concatenate([[0.0], position[:-1]])
    trade = np.abs(position - prev)
    r = position * actual - cost_rate * trade
    ret = (np.exp(np.sum(r)) - 1) * 100
    sd  = np.std(r)
    sh  = 0.0 if sd == 0 else np.mean(r) / sd * np.sqrt(252)
    return ret, sh, int((trade > 1e-12).sum())


def evaluate(actual, pred, position=None):
    """One result row with all metrics. position=None -> long-only: 1 if pred > 0 else 0."""
    mse, rmse, mae, da = accuracy_metrics(actual, pred)
    if position is None:
        position = (pred > 0).astype(float)
    rg, sg, nt = strategy_with_position(actual, position, 0.0)
    rb, sb, _  = strategy_with_position(actual, position, COST_BASE)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "DA": da,
            "Ret_gross_%": rg, "Sharpe_gross": sg, "N_Trades": nt,
            "Ret_net_0.15_%": rb, "Sharpe_net_0.15": sb}


# ----------------------------------------------------------------------------
# Mean +/- std helpers (sample std, ddof = 1)
# ----------------------------------------------------------------------------
def fmt_value(x, col):
    return "—" if pd.isna(x) else f"{x:.{DECIMALS[col]}f}"


def mean_std_tables(per_seed, deterministic=()):
    """per_seed: DataFrame with columns Model, Seed, <metrics>.
    Returns (text_table 'mean ± std', numeric_table with <col>_mean/<col>_std, mean_only_table).
    Models listed in `deterministic`, or whose std is 0 for every metric, are shown as a single value."""
    txt, num, mean_only = [], [], []
    for model, g in per_seed.groupby("Model", sort=False):
        rt, rn, rm = {"Model": model}, {"Model": model}, {"Model": model}
        sds = {c: (g[c].std(ddof=1) if len(g) > 1 else 0.0) for c in METRIC_COLS if c in g}
        det = model in deterministic or all((np.nan_to_num(s) < 1e-12) for s in sds.values())
        for c in METRIC_COLS:
            if c not in g:
                continue
            mu, sd = g[c].mean(), np.nan_to_num(sds[c])
            rt[c] = fmt_value(mu, c) if (det or pd.isna(mu)) else f"{fmt_value(mu, c)} ± {fmt_value(sd, c)}"
            rn[c + "_mean"], rn[c + "_std"] = mu, sd
            rm[c] = mu
        txt.append(rt); num.append(rn); mean_only.append(rm)
    return pd.DataFrame(txt), pd.DataFrame(num), pd.DataFrame(mean_only)


def round_mean_only(df):
    out = df.copy()
    for c in METRIC_COLS:
        if c in out:
            out[c] = out[c].round(DECIMALS[c])
            if c == "N_Trades":
                out[c] = out[c].round(0).astype("Int64")
    return out


# ----------------------------------------------------------------------------
# The proposed model (AG-LSTM, AG-GRU, AG-LSTM-GRU) for ONE stock and ONE seed
# ----------------------------------------------------------------------------
def run_proposed(stock, seed):
    """ARIMAX -> GARCH -> DL (with GARCH volatility input) for one stock and one seed.
    Writes results/Results_proposed_<S>_seed<seed>.csv and
           results/Predictions_<S>_proposed_seed<seed>.csv  (used by dm_test.py)."""
    P = Prepared(stock)
    set_seed(seed)
    ar = fit_arimax(P)
    gv = fit_garch(P, ar)
    data = make_dl_data(P, ar, gv)
    rows, preds = [], {}
    for mt in DL_TYPES:
        model = build_and_train_dl(mt, P.cfg, data, seed)
        pred  = dl_forecast(model, data, ar["arimax_return"])
        name  = AG_NAMES[mt]
        preds[name] = pred
        rows.append({"Stock": stock, "Seed": seed, "Model": name, **evaluate(P.actual_return, pred)})
        print(f"  seed {seed}: {name} done")
    res = pd.DataFrame(rows).round(6)
    print(res.to_string(index=False))
    res.to_csv(out_path(f"Results_proposed_{stock}_seed{seed}.csv"), index=False)
    pd.DataFrame({"Date": P.pred_dates, "Actual": P.actual_return, **preds}).to_csv(
        out_path(f"Predictions_{stock}_proposed_seed{seed}.csv"), index=False)
    return res


def proposed_main(seed):
    """Entry point shared by proposed_seed42.py / proposed_seed77.py / proposed_seed123.py."""
    for s in stocks_from_argv():
        print("=" * 70); print(f"PROPOSED MODEL - {s} - seed {seed}"); print("=" * 70)
        run_proposed(s, seed)
