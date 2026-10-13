# -*- coding: utf-8 -*-
"""
Proposed model (AG-LSTM, AG-GRU, AG-LSTM-GRU) - SEED 42 only.
ARIMAX -> GARCH -> deep learning (lagged residual + market features + GARCH volatility).
Long-only strategy with transaction costs (0 / 0.15 %).

Usage:   python src/proposed_seed42.py MASR      (or CCAP / PHDC / HELI / ALL)
Output:  results/Results_proposed_<STOCK>_seed42.csv
         results/Predictions_<STOCK>_proposed_seed42.csv   (forecasts, used by dm_test.py)
Run proposed_seed42.py, proposed_seed77.py and proposed_seed123.py, then proposed_mean_std.py.
"""
from common import proposed_main

SEED = 42            # fixed a priori (42, 77, 123)

if __name__ == "__main__":
    proposed_main(SEED)
