import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBRegressor

class SpreadSignalModel:
    def __init__(
        self,
        mas=[5, 10, 15],
        emas=[10],
        rsis=[10],
        macds=[(12, 26)],
        hmm_n_components=2,
        random_state=101,
    ):
        self.mas = mas
        self.emas = emas
        self.rsis = rsis
        self.macds = macds
        self.hmm_n_components = hmm_n_components
        self.random_state = random_state

    def get_spread_signal(self, spread: pd.DataFrame) -> int:
        # === HMM STATE PREDICTION === #
        hmm_model = GaussianHMM(n_components=self.hmm_n_components, random_state=self.random_state)
        hmm_model.fit(spread)
        states = hmm_model.predict(spread)
        state_df = pd.DataFrame(states, columns=["state"], index=spread.index)

        # === FEATURE ENGINEERING === #
        lag = spread.shift(1)
        features = pd.DataFrame(index=spread.index)
        features["lag"] = lag
        features["lag_state"] = state_df.shift(1)["state"]

        # MAs and EMAs
        for period in self.mas:
            features[f"ma_{period}"] = lag.rolling(window=period).mean()
        for period in self.emas:
            features[f"ema_{period}"] = lag.ewm(span=period, adjust=False).mean()

        # RSI
        for period in self.rsis:
            delta = lag.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        for short, long in self.macds:
            ema_short = lag.ewm(span=short, adjust=False).mean()
            ema_long = lag.ewm(span=long, adjust=False).mean()
            features[f"macd_{short}_{long}"] = ema_short - ema_long

        features.dropna(inplace=True)
        aligned_spread = spread.loc[features.index]

        # === XGBOOST MODEL === #
        xgb_model = XGBRegressor(
            learning_rate=0.4,
            n_estimators=400,
            max_depth=5,
            gamma=0.005,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=2
        )

        xgb_model.fit(
            features.iloc[:-1],
            aligned_spread.iloc[:-1].values.ravel(),
            eval_set=[(features.iloc[:-1], aligned_spread.iloc[:-1].values.ravel())],
            verbose=False
        )

        predicted = xgb_model.predict(features.iloc[[-1]])[0]
        last_actual = aligned_spread.iloc[-1].values[0]

        return int(np.sign(predicted - last_actual))