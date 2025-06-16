import datetime as dt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

class Cluster:
    def __init__(self, data_handler, features=None, use_pca=False, seed=42):
        self.data_handler = data_handler
        self.features = features or []
        self.use_pca = use_pca
        self.seed = seed

    def get_clusters(self, date=dt.date.today(), window=252):
        data_window = (
            self.data_handler.get_data_window(date=date, window=window)
            .ffill()
            .dropna(axis=1)
        )

        returns = np.log(data_window).diff().dropna()

        scaler = StandardScaler()

        if not self.features:
            features = pd.DataFrame(
                scaler.fit_transform(returns.T),
                columns=returns.T.columns,
                index=returns.T.index
            )
            _features = features
        else:
            stats = pd.DataFrame(index=returns.columns)
            stats["mean"] = returns.mean() * window
            stats["variance"] = returns.var() * window
            stats["skewness"] = returns.skew()
            stats["kurtosis"] = returns.kurtosis()

            features = pd.DataFrame(
                scaler.fit_transform(stats),
                columns=stats.columns,
                index=stats.index
            )
            _features = features[self.features]

        if self.use_pca:
            pca = PCA(n_components="mle", random_state=self.seed)
            pca.fit(_features)
            features = pd.DataFrame(
                pca.transform(_features),
                index=_features.index
            )

        ap = AffinityPropagation(random_state=self.seed)
        ap.fit(_features)
        clusters = pd.Series(
            ap.predict(_features),
            index=_features.index
        )

        return clusters, data_window, _features