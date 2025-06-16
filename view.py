import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# PAIRS_DIR = "data/cache/pairs"
# cache_file = os.path.join(PAIRS_DIR, f"2020-12-31.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_pairs = cached["pairs"]
#
#     for idx, pair in enumerate(cached_pairs):
#         print(idx, pair[0], pair[1], pair[2])
#
# CLUSTER_DIR = "data/cache/clusters"
# cache_file = os.path.join(CLUSTER_DIR, f"2020-12-31.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_clusters = cached["clusters"]
#     data_window = cached["data_window"]
    # for idx, clusters in enumerate(zip(cached_clusters, data_window.columns)):
    #     print(idx, clusters)

# SIGNAL_DIR = "data/cache/signal"
# cache_file = os.path.join(SIGNAL_DIR, f"2021-04-15.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_spread = cached["spread"]
#     cached_signal = cached["signal"]
#     print(cached_spread)
#     print(cached_signal)

# PNL_DIR = "data/cache/pnl"
# cache_file = os.path.join(PNL_DIR, f"2021-01-04.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_pnl = cached["pnl"]
#     print(cached_pnl)

pnl_df = pd.read_csv("data/full_pnl.csv", index_col=0, parse_dates=True).sort_index()
print(pnl_df.head())
daily_pnl = pnl_df.sum(axis=1)  # total return per day
cumulative_pnl = daily_pnl.cumsum()/1000000
print(cumulative_pnl.head())
print(cumulative_pnl.min())
# non_zero_counts = pnl_df.astype(bool).sum(axis=1)


# # Plotting
cumulative_pnl.plot(title="Cumulative PnL ($1,000,000 initial investment)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (%)")
plt.grid(True)
plt.show()