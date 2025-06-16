import os
import pickle
#
# PAIRS_DIR = "data/cache/pairs"
# cache_file = os.path.join(PAIRS_DIR, f"2022-05-03.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_pairs = cached["pairs"]
#
#     for idx, pair in enumerate(cached_pairs):
#         print(idx, pair[0], pair[1])
#
# CLUSTER_DIR = "data/cache/clusters"
# cache_file = os.path.join(CLUSTER_DIR, f"2022-05-03.pkl")
# with open(cache_file, "rb") as f:
#     cached = pickle.load(f)
#     cached_clusters = cached["clusters"]
#     data_window = cached["data_window"]
    # for idx, clusters in enumerate(zip(cached_clusters, data_window.columns)):
    #     print(idx, clusters)

SIGNAL_DIR = "data/cache/signal"
cache_file = os.path.join(SIGNAL_DIR, f"2020-12-31.pkl")
with open(cache_file, "rb") as f:
    cached = pickle.load(f)
    cached_spread = cached["spread"]
    cached_signal = cached["signal"]
    print(cached_spread)
    print(cached_signal)