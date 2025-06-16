import datetime as dt
import pandas as pd

class DataHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_data_window(self, date=dt.date.today(), window=252):
        idx = self.data.index.get_loc(self.data.index[self.data.index <= pd.to_datetime(date)][-1])
        data_window = self.data.iloc[idx - window: idx]
        return data_window