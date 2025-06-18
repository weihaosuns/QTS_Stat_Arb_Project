import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PAIRS_DIR = "data/cache/pairs"
# cache_file = os.path.join(PAIRS_DIR, f"2023-05-24.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_pairs = cached["pairs"]
#
#     for idx, pair in enumerate(cached_pairs):
#         print(idx, pair[0], pair[1], pair[2])
#
# CLUSTER_DIR = "data/cache/clusters"
# cache_file = os.path.join(CLUSTER_DIR, f"2023-05-24.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_clusters = cached["clusters"]
#     data_window = cached["data_window"]
#     for idx, clusters in enumerate(zip(cached_clusters, data_window.columns)):
#         print(idx, clusters)

SIGNAL_DIR = "data/cache/signal"
cache_file = os.path.join(SIGNAL_DIR, f"2023-05-24.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_spread = cached["spread"]
#     cached_signal = cached["signal"]
#     spreads = cached_spread
#
#     returns_dict = {}
#
#     for pair, df in spreads.items():
#         df = df.copy()
#         df = df.sort_index()
#         df['return'] = df['spread'].pct_change()
#         returns_dict[pair] = df['return']
#
#     spread_returns = pd.DataFrame(returns_dict).dropna()
#     mu = spread_returns.mean()
#     sigma = spread_returns.cov()
#
#     inv_Sigma = np.linalg.inv(sigma)
#     raw_weights = inv_Sigma @ mu
#     weights = raw_weights / np.sum(np.abs(raw_weights))
#     optimal_weights = pd.Series(weights, index=spread_returns.columns)
#
#     print(optimal_weights)
#     print(optimal_weights["ACN_ATO"])

PNL_DIR = "data/cache/pnl"
cache_file = os.path.join(PNL_DIR, f"2023-05-24.pkl")
with open(cache_file, "rb") as f:
    cached = pickle.load(f)
    cached_pnl = cached["pnl"]
    print(cached_pnl)

pnl_df = pd.read_csv("data/full_pnl.csv", index_col=0, parse_dates=True).sort_index()
print(pnl_df.head())
daily_pnl = pnl_df.sum(axis=1)  # total return per day
cumulative_pnl = daily_pnl.cumsum()
# print(daily_pnl[daily_pnl > 500000])
# print(cumulative_pnl.min())
# non_zero_counts = pnl_df.astype(bool).sum(axis=1)

# 2021-04-14    5.465363e+05
# 2021-04-21    4.337580e+06
# 2021-04-30    2.239333e+06
# 2021-05-07    8.442578e+05
# 2021-07-06    7.695042e+05
# 2021-07-29    6.719224e+06
# 2022-04-06    2.489486e+06
# 2022-08-19    1.283245e+06
# 2022-11-28    1.454334e+06
# 2023-05-24    1.085088e+07
# 2023-12-15    5.036718e+05

# Plotting
cumulative_pnl.plot(title="Cumulative PnL (%)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (%)")
plt.grid(True)
plt.show()