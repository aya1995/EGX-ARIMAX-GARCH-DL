# -*- coding: utf-8 -*-
"""
Proposed model: mean +/- std over the three seeds (42, 77, 123).
Reads  results/Results_proposed_<STOCK>_seed{42,77,123}.csv  (written by proposed_seed*.py).
Std = sample standard deviation (ddof = 1).

Usage:   python src/proposed_mean_std.py          (all stocks found)   |   python src/proposed_mean_std.py MASR
Output (per stock S):
  results/Proposed_per_seed_<S>.csv        all seeds stacked
  results/Proposed_mean_std_<S>.csv        paper table, text "mean ± std" (rounded)
  results/Proposed_mean_numeric_<S>.csv    numeric means only (input of friedman_finner.py)
  results/Proposed_std_numeric_<S>.csv     numeric std (ddof=1)
  results/Proposed_mean_of_3_models.csv    all stocks: average of AG-LSTM / AG-GRU / AG-LSTM-GRU (for the ablation table)
"""
import pandas as pd
from common import (SEEDS, STOCKS, METRIC_COLS, DECIMALS, stocks_from_argv, out_path,
                    mean_std_tables, round_mean_only)

summary_rows = []
for s in stocks_from_argv():
    frames = []
    for sd in SEEDS:
        try:
            frames.append(pd.read_csv(out_path(f"Results_proposed_{s}_seed{sd}.csv")))
        except FileNotFoundError:
            print(f"[{s}] missing Results_proposed_{s}_seed{sd}.csv - run proposed_seed{sd}.py first")
    if len(frames) != len(SEEDS):
        print(f"[{s}] skipped (need all seeds {SEEDS})"); continue
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(out_path(f"Proposed_per_seed_{s}.csv"), index=False)

    txt, num, mean_only = mean_std_tables(per)
    txt.to_csv(out_path(f"Proposed_mean_std_{s}.csv"), index=False, encoding="utf-8-sig")
    mean_only.to_csv(out_path(f"Proposed_mean_numeric_{s}.csv"), index=False)
    num[["Model"] + [c + "_std" for c in METRIC_COLS]].to_csv(out_path(f"Proposed_std_numeric_{s}.csv"), index=False)

    print("\n" + "=" * 70); print(f"{s}: mean ± std over seeds {SEEDS}"); print("=" * 70)
    print(txt.to_string(index=False))

    m3 = mean_only[METRIC_COLS].mean()          # average of the three proposed models
    summary_rows.append({"Stock": s, "Model": "ARIMAX+GARCH+DL (mean of 3)", **m3.to_dict()})

if summary_rows:
    out = round_mean_only(pd.DataFrame(summary_rows))
    out.to_csv(out_path("Proposed_mean_of_3_models.csv"), index=False)
    print("\nAverage of the three proposed models per stock:")
    print(out.to_string(index=False))
