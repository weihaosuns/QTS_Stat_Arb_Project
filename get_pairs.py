import datetime as dt
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression

class PairSelector:
    def __init__(self, significance_coint=0.05, significance_adf=0.05):
        self.significance_coint = significance_coint
        self.significance_adf = significance_adf

    def get_pairs(self, date=dt.date.today(), window=252, keyed=False,
                  clusters=None, data_window=None, features=None):

        def test_pairs_coint_adf(data):
            pairs = []
            keys = data.columns
            X = data.values  # speedup: work with NumPy arrays
            n = X.shape[1]

            for i in range(n):
                for j in range(i + 1, n):
                    a = X[:, i].reshape(-1, 1)
                    b = X[:, j].reshape(-1, 1)

                    # Cointegration test
                    score_coint, pvalue_coint, _ = coint(a, b)
                    if pvalue_coint < self.significance_coint:
                        # OLS
                        model = LinearRegression()
                        model.fit(b, a)
                        beta = model.coef_[0][0]
                        score_ols = model.score(b, a)

                        # ADF on residuals
                        residuals = a - model.predict(b)
                        pvalue_adf = adfuller(residuals.ravel())[1]

                        if pvalue_adf < self.significance_adf:
                            pairs.append((keys[i], keys[j], beta, score_coint, pvalue_coint, score_ols, pvalue_adf))

            return pairs

        if keyed:
            pairs = {}
        else:
            pairs = []

        # Cache cluster labels
        cluster_labels = clusters.value_counts().index

        for i, cluster_label in enumerate(cluster_labels):
            tickers = clusters[clusters == cluster_label].index
            sub_data = data_window.loc[:, tickers]
            if sub_data.shape[1] < 2:
                continue  # skip 1-ticker clusters

            _pairs = test_pairs_coint_adf(sub_data)

            if keyed:
                pairs[i] = _pairs
            else:
                pairs.extend(_pairs)

        return pairs, data_window