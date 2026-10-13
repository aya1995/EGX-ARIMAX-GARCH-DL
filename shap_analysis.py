# -*- coding: utf-8 -*-
"""
SHAP Analysis - Final Return Prediction
✅ SHAP على الـ Final Return (ARIMAX + DL)
✅ DL features + EMA_10
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)

FILE_NAME   = "PHDC_features_usd_egp.csv"
TRAIN_RATIO = 0.80
SL          = 20
EPOCHS      = 100
PATIENCE    = 15

# Table 3.5 - parameters of the model used here (AG-LSTM-GRU only), selected on the training split
STOCK = FILE_NAME.split("_")[0]
TABLE_3_5 = {   # ARIMAX order, GARCH (p,q), LSTM-GRU (LSTM units, GRU units), dropout, batch size
    "MASR": {"arimax": (1,0,0), "garch": (1,1), "units": (32,32), "dropout": 0.1, "batch": 64},
    "CCAP": {"arimax": (3,0,0), "garch": (1,2), "units": (32,32), "dropout": 0.1, "batch": 64},
    "PHDC": {"arimax": (1,0,1), "garch": (1,2), "units": (32,32), "dropout": 0.2, "batch": 32},
    "HELI": {"arimax": (1,0,0), "garch": (2,2), "units": (32,32), "dropout": 0.1, "batch": 64},
}
CFG = TABLE_3_5[STOCK]

# =========================
# 1. Load + Features
# =========================
print("--- Loading Data ---")
df = pd.read_csv(FILE_NAME)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

df["Log_Return"]    = np.log(df["Close"]/df["Close"].shift(1))
df["Target_Return"] = df["Log_Return"].shift(-1)
df["Open_return"]   = np.log(df["Open"]/df["Open"].shift(1))
df["USD_change"]    = df["USD_EGP"].pct_change()*100
df["Volume_change"] = df["Volume"].pct_change()*100
df["HL_range_pct"]  = (df["High"]-df["Low"])/(df["Close"]+1e-10)*100

ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
df["EMA_10"]        = ema_10/(df["Close"]+1e-10)         # ✅ normalized

ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]          = (ema_12 - ema_26)/(df["Close"]+1e-10)  # normalized
df["MACD_signal"]   = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_sig"]      = df["MACD_signal"]/(df["Close"]+1e-10)

pc = df["Close"].shift(1)
tr = pd.concat([df["High"]-df["Low"],
                (df["High"]-pc).abs(),
                (df["Low"]-pc).abs()], axis=1).max(axis=1)
df["ATR"]        = tr.rolling(14).mean()/(df["Close"]+1e-10)*100
l14 = df["Low"].rolling(14).min()
h14 = df["High"].rolling(14).max()
df["Stochastic"] = 100*((df["Close"]-l14)/(h14-l14+1e-10))
df["Return_5D"]  = np.log(df["Close"]/df["Close"].shift(5))
df = df.replace([np.inf,-np.inf],np.nan).dropna().copy()

n = int(len(df)*TRAIN_RATIO)
train_df = df.iloc[:n].copy()
test_df  = df.iloc[n:].copy()
train_target = train_df["Target_Return"].values.astype(float)
test_target  = test_df["Target_Return"].values.astype(float)

arimax_features = ["Open_return","HL_range_pct","Volume_change","USD_change"]

# ✅ DL features + EMA_10
dl_features = ["Return_5D","EMA_10",
               "MACD","MACD_sig","ATR","Stochastic"]

train_exog   = train_df[arimax_features].values.astype(float)
test_exog    = test_df[arimax_features].values.astype(float)
train_market = train_df[dl_features].values.astype(float)
test_market  = test_df[dl_features].values.astype(float)

print(f"DL features ({len(dl_features)}): {dl_features}")

# =========================
# 2. ARIMAX + GARCH
# =========================
try:
    af = ARIMA(train_target, exog=train_exog, order=CFG["arimax"]).fit()
except Exception as e:
    print(f"ARIMAX{CFG['arimax']} failed ({e}); using (1,0,0)")
    af = ARIMA(train_target, exog=train_exog, order=(1,0,0)).fit()
train_ap    = np.asarray(af.predict(start=0, end=len(train_target)-1,
                                     exog=train_exog)).flatten()
test_ap     = np.asarray(af.forecast(steps=len(test_target),
                                      exog=test_exog)).flatten()
train_resid = train_target - train_ap
test_resid  = test_target  - test_ap

G_P, G_Q = CFG["garch"]
gf = arch_model(train_resid, mean="Zero", vol="GARCH",
                p=G_P, q=G_Q, rescale=False).fit(disp="off")
tr_gvol = np.asarray(gf.conditional_volatility).flatten()
prm = gf.params
om = float(prm.get("omega",0))
al = [float(prm.get(f"alpha[{i}]",0)) for i in range(1,G_P+1)]
be = [float(prm.get(f"beta[{i}]",0)) for i in range(1,G_Q+1)]
eps2 = list(train_resid[::-1][:G_P]**2)      # most recent first
var  = list(tr_gvol[::-1][:G_Q]**2)
te_gvol=[]
for _ in range(len(test_target)):
    nv = om + sum(al[i]*eps2[i] for i in range(G_P)) + sum(be[j]*var[j] for j in range(G_Q))
    nv = max(nv,1e-12)
    te_gvol.append(np.sqrt(nv))
    eps2 = [nv]+eps2[:G_P-1]; var = [nv]+var[:G_Q-1]
te_gvol=np.array(te_gvol)

sc_r=StandardScaler(); tr_r=sc_r.fit_transform(train_resid.reshape(-1,1)); te_r=sc_r.transform(test_resid.reshape(-1,1))
sc_m=StandardScaler(); tr_m=sc_m.fit_transform(train_market); te_m=sc_m.transform(test_market)
sc_g=StandardScaler(); tr_g=sc_g.fit_transform(tr_gvol.reshape(-1,1)); te_g=sc_g.transform(te_gvol.reshape(-1,1))
tr_rh=np.roll(tr_r,1,axis=0); tr_rh[0]=0.0
te_rh=np.zeros_like(te_r); te_rh[0]=tr_r[-1]
tr_dl=np.concatenate([tr_rh,tr_m,tr_g],axis=1)
te_dl=np.concatenate([te_rh,te_m,te_g],axis=1)
print(f"DL input shape: {tr_dl.shape[1]} features")

def seqs(f,t,sl):
    X,y=[],[]
    for i in range(sl,len(f)):
        X.append(f[i-sl:i]); y.append(t[i])
    return np.array(X),np.array(y)

Xtr,ytr=seqs(tr_dl,tr_r.flatten(),SL)
Xte,_ =seqs(te_dl,te_r.flatten(),SL)
arimax_return_test = test_ap[SL:]

# =========================
# 3. Train AG-LSTM-GRU (Best Model)
# =========================
print("--- Training AG-LSTM-GRU ---")
shape = (Xtr.shape[1], Xtr.shape[2])
best_model_name = 'LSTM-GRU'

tf.random.set_seed(SEED)
model = Sequential([LSTM(CFG["units"][0], return_sequences=True, input_shape=shape),
                    GRU(CFG["units"][1]), Dropout(CFG["dropout"]), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS,
          batch_size=CFG["batch"], verbose=0, shuffle=False,
          callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                   restore_best_weights=True)])
print(f"✅ AG-LSTM-GRU trained")

# =========================
# 4. SHAP on Final Return
# =========================
print("\n--- Computing SHAP ---")

Xte_flat = Xte.reshape(Xte.shape[0], -1)
Xtr_flat = Xtr.reshape(Xtr.shape[0], -1)
n_features_flat = Xtr_flat.shape[1]
n_orig = tr_dl.shape[1]

def predict_final_return(X_flat):
    X_3d    = X_flat.reshape(-1, SL, n_orig)
    dl_pred = model.predict(X_3d, verbose=0).flatten()
    dl_ret  = sc_r.inverse_transform(dl_pred.reshape(-1,1)).flatten()
    return dl_ret  # ← DL correction only

n_background = min(100, len(Xtr_flat))
background   = shap.sample(Xtr_flat, n_background, random_state=SEED)
explainer    = shap.KernelExplainer(predict_final_return, background)
n_explain    = min(200, len(Xte_flat))
shap_values  = explainer.shap_values(Xte_flat[:n_explain], nsamples=100)

# =========================
# 5. Feature Importance
# ✅ اجمع SHAP per original feature (average across all lags)
# =========================
feature_labels = (
    ["ARIMAX Residual (lag)"] +
    [f"{f} (lag)" for f in dl_features] +
    ["GARCH Volatility (lag)"]
)

# ✅ اسم مميز لكل lag
short_names = (
    [f"Resid_t-{SL-i}" for i in range(SL)] +
    [f"{f}_t-{SL-i}" for f in dl_features for i in range(SL)] +
    [f"GARCH_t-{SL-i}" for i in range(SL)]
)

shap_abs   = np.abs(shap_values)
importance = np.zeros(n_orig)
for i in range(n_orig):
    cols = list(range(i, n_features_flat, n_orig))
    importance[i] = shap_abs[:, cols].mean()

imp_df = pd.DataFrame({'Feature':feature_labels,
                        'Mean_SHAP':importance}).sort_values('Mean_SHAP', ascending=False)
print("\n📊 Feature Importance (SHAP - averaged across lags):")
print(imp_df.to_string(index=False))
imp_df.to_csv('SHAP_FinalReturn_Importance.csv', index=False)

# ✅ Bar chart: importance per original feature
plt.figure(figsize=(10,6))
plt.barh(imp_df['Feature'][::-1], imp_df['Mean_SHAP'][::-1],
         color='darkorange', edgecolor='black', alpha=0.8)
plt.xlabel('Mean |SHAP Value| (averaged across lags)')
plt.title(f'SHAP Feature Importance\n({best_model_name} - Final Return)',
          fontweight='bold', fontsize=13)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('SHAP_FinalReturn_Importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ✅ Aggregate SHAP per original feature (mean across all lags)
shap_per_feature = np.zeros((n_explain, n_orig))
X_per_feature    = np.zeros((n_explain, n_orig))
for i in range(n_orig):
    cols = list(range(i, n_features_flat, n_orig))
    shap_per_feature[:, i] = shap_values[:, cols].mean(axis=1)
    X_per_feature[:, i]    = Xte_flat[:n_explain, cols].mean(axis=1)

## =========================
# SHAP Summary Plot - Improved Layout
# =========================

plt.figure(figsize=(11, 7))

shap.summary_plot(
    shap_per_feature,
    X_per_feature,
    feature_names=feature_labels,
    show=False,
    max_display=n_orig
)

plt.title(
    f'SHAP Summary - {best_model_name} Final Return',
    fontweight='bold',
    fontsize=14,
    pad=15
)

plt.xlabel(
    'SHAP Value (Impact on Model Output)',
    fontsize=11,
    labelpad=10
)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout(pad=2)

plt.savefig(
    'SHAP_FinalReturn_Summary.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"\n✅ Done! Best Model: {best_model_name}")
print(f"   ✅ DL features: {dl_features}")
print(f"   ✅ SHAP_FinalReturn_Importance.csv")