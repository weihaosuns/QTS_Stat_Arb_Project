import os
import pickle
from tqdm import tqdm
from get_data import DataHandler
from get_clusters import Cluster
from get_pairs import PairSelector
from get_signal import SpreadSignalModel
import pandas as pd
import numpy as np

DATA_PATH = "data/data_eq.csv"
CLUSTER_DIR = "data/cache/clusters"
WINDOW = 252
SEED = 42

os.makedirs(CLUSTER_DIR, exist_ok=True)

# Load data
data = pd.read_csv(DATA_PATH,
                   index_col=0,
                   parse_dates=True
                   )

# Roll over all valid end dates
train_dates = data.index[WINDOW: int(-len(data.index)*0.2)]
test_dates = data.index[int(len(data.index)*0.2): ]

# Initialize Classes
data_handler = DataHandler(data)

# Initialize clustering class (from your get_clusters.py)
cluster_model = Cluster(
    data_handler=data_handler,
    features=["mean", "variance", "skewness", "kurtosis"],  # or [] to use raw returns
    use_pca=False,
    seed=42
)


# for date in tqdm(train_dates):
#     cache_file = os.path.join(CLUSTER_DIR, f"{date.date()}.pkl")
#     if os.path.exists(cache_file):
#         continue  # Skip if already cached
#
#     try:
#         clusters, data_window, features = cluster_model.get_clusters(date=date, window=WINDOW)
#         cache_data = {
#             "date": date,
#             "clusters": clusters,
#             "features": features,
#             "data_window": data_window,
#         }
#         with open(cache_file, 'wb') as f:
#             pickle.dump(cache_data, f)
#     except Exception as e:
#         print(f"Failed on {date}: {e}")

    # AP did not converge on 517 and 652  !!!

from joblib import Parallel, delayed

PAIRS_DIR = "data/cache/pairs"
os.makedirs(PAIRS_DIR, exist_ok=True)
selector = PairSelector()

# def cache_pairs(date):
#     cache_file = os.path.join(PAIRS_DIR, f"{date.date()}.pkl")
#     if os.path.exists(cache_file):
#         return f"Skipped {date.date()}"
#
#     try:
#         cluster_path = os.path.join(CLUSTER_DIR, f"{date.date()}.pkl")
#         if not os.path.exists(cluster_path):
#             return f"No cluster file for {date.date()}"
#
#         with open(cluster_path, "rb") as f:
#             cached = pickle.load(f)
#
#         clusters = cached["clusters"]
#         features = cached["features"]
#         data_window = cached["data_window"]
#
#         pairs, data_window = selector.get_pairs(date=date, window=WINDOW, keyed=False,
#                   clusters=clusters, data_window=data_window, features=features)
#         cache_data = {
#             "date": date,
#             "pairs": pairs,
#             "data_window": data_window,
#         }
#
#         with open(cache_file, 'wb') as f:
#             pickle.dump(cache_data, f)
#         return f"Done {date.date()}"
#
#     except Exception as e:
#         return f"Failed {date.date()}: {e}"
#
# results = Parallel(n_jobs=-1)(
#     delayed(cache_pairs)(date) for date in tqdm(train_dates)
# )
#
# # Log results
# for r in results:
#     print(r)


SIGNAL_DIR = "data/cache/signal"
os.makedirs(SIGNAL_DIR, exist_ok=True)
signaler = SpreadSignalModel()
#
# def cache_signal(date):
#     cache_file = os.path.join(SIGNAL_DIR, f"{date.date()}.pkl")
#     if os.path.exists(cache_file):
#         return f"Skipped {date.date()}"
#
#     try:
#         pairs_path = os.path.join(PAIRS_DIR, f"{date.date()}.pkl")
#         if not os.path.exists(pairs_path):
#             return f"No cluster file for {date.date()}"
#
#         with open(pairs_path, "rb") as f:
#             cached = pickle.load(f)
#
#         pairs = cached["pairs"]
#         data_window = cached["data_window"]
#
#         signals = {}
#         spreads = {}
#
#         for pair in pairs:
#             a = data_window[pair[0]]
#             b = data_window[pair[1]]
#             beta = pair[2]
#             spread = a - beta * b
#             spread = spread.to_frame(name="spread")
#             signal = signaler.get_spread_signal(spread)
#             pair_name = f"{pair[0]}_{pair[1]}"
#             signals[pair_name] = signal
#             spreads[pair_name] = spread
#
#         cache_data = {
#             "date": date,
#             "spread": spreads,
#             "signal": signals,
#             "data_window": data_window,
#         }
#
#         with open(cache_file, 'wb') as f:
#             pickle.dump(cache_data, f)
#         return f"Done {date.date()}"
#
#     except Exception as e:
#         return f"Failed {date.date()}: {e}"
#
# results = Parallel(n_jobs=-1)(
#     delayed(cache_signal)(date) for date in tqdm(train_dates)
# )
#
# # Log results
# for r in results:
#     print(r)

# NUMBER 96, 512, 608, 624 HMM MODEL IS NOT CONVERGING.

PNL_DIR = "data/cache/pnl"
os.makedirs(PNL_DIR, exist_ok=True)
price_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

notional = 1000000

for i in tqdm(range(len(train_dates) - 1)):
    date = train_dates[i]
    next_date = train_dates[i + 1]

    pnl_cache_file = os.path.join(PNL_DIR, f"{next_date.date()}.pkl")
    # if os.path.exists(pnl_cache_file):
    #     continue  # skip if done

    try:
        pairs_path = os.path.join(PAIRS_DIR, f"{date.date()}.pkl")
        with open(pairs_path, "rb") as f:
            pairs_cache = pickle.load(f)
        pairs = pairs_cache["pairs"]

        signal_path = os.path.join(SIGNAL_DIR, f"{date.date()}.pkl")
        with open(signal_path, "rb") as f:
            signal_cache = pickle.load(f)
        signals = signal_cache["signal"]

        pnl_dict = {}
        notional_per_pair = notional/2 /len(pairs)
        for pair in pairs:
            ticker_a, ticker_b, beta = pair[0], pair[1], pair[2]
            pair_name = f"{ticker_a}_{ticker_b}"

            if pair_name not in signals:
                continue
            signal = signals[pair_name]

            if (ticker_a not in price_df.columns) or (ticker_b not in price_df.columns):
                continue
            if (date not in price_df.index) or (next_date not in price_df.index):
                continue

            price_a_t = price_df.at[date, ticker_a]
            price_b_t = price_df.at[date, ticker_b]
            price_a_t1 = price_df.at[next_date, ticker_a]
            price_b_t1 = price_df.at[next_date, ticker_b]

            if any(np.isnan([price_a_t, price_b_t, price_a_t1, price_b_t1])):
                continue

            spread_t = price_a_t - beta * price_b_t
            spread_t1 = price_a_t1 - beta * price_b_t1

            if abs(spread_t) < 1e-6:
                continue

            spread_return = (spread_t1 - spread_t) / abs(spread_t)

            # max loss is the entire trade, max gain is unlimited.
            pnl = np.clip(signal * spread_return * notional_per_pair, -notional_per_pair, np.inf)
            pnl_dict[pair_name] = pnl

        with open(pnl_cache_file, "wb") as f:
            pickle.dump({
                "date": next_date,
                "pnl": pnl_dict,
            }, f)

    except Exception as e:
        print(f"Failed on {date}: {e}")

PNL_DIR = "data/cache/pnl"

# List and sort all PnL pickle files
pnl_files = sorted([
    f for f in os.listdir(PNL_DIR) if f.endswith(".pkl")
])

# Load into a list of DataFrames
pnl_rows = []

for fname in tqdm(pnl_files):
    path = os.path.join(PNL_DIR, fname)
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        date = pd.to_datetime(cache["date"])
        pnl = cache["pnl"]  # dict or pd.Series
        row = pd.Series(pnl, name=date)
        pnl_rows.append(row)
    except Exception as e:
        print(f"Failed to load {fname}: {e}")

# Combine into full PnL DataFrame
pnl_df = pd.DataFrame(pnl_rows).sort_index()

# Fill NaNs with 0 if you assume no position = 0 PnL
pnl_df = pnl_df.fillna(0)

# Save for future use
pnl_df.to_csv("data/full_pnl.csv")