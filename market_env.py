import pandas as pd
import numpy as np


class MarketEnv:
    def __init__(self):
        self.data = pd.read_csv('./data/Quant_Invest_Fundusze.csv', sep=';', index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        self.data /= self.data.iloc[0, :]

        self.month_to_month = self.data.groupby(pd.Grouper(freq='M'), as_index=False).nth(0).pct_change() + 1
        self.month_to_month.iloc[0] = self.data.iloc[0]

        self.training_period_start = None
        self.training_period_end = None

        # those are indices of months in self.month_to_month
        self.training_start_month = None
        self.current_month = None
        self.training_end_month = None

        self._transaction_cost = 0

    def set_training_period(self, start, end):
        self.training_period_start = pd.to_datetime(start).replace(day=1)
        self.training_period_end = pd.to_datetime(end).replace(day=1)

        if self.training_period_start >= self.training_period_end:
            raise ValueError('Training period start cannot be same as or later than end')

        self.training_start_month = self.__months_diff(self.data.index[0], self.training_period_start)
        self.training_end_month = self.__months_diff(self.data.index[0], self.training_period_end)

        self.reset()

    def reset(self):
        self.current_month = self.training_start_month

    def step(self):
        self.current_month += 1

    def progress(self):
        return (self.current_month - self.training_start_month) / (self.training_end_month - self.training_start_month)

    def current_mtm_returns(self):
        return self.month_to_month.iloc[self.current_month].values

    def next_mtm_returns(self):
        return self.month_to_month.iloc[self.current_month + 1].values

    def should_continue(self):
        return self.current_month < self.training_end_month

    def number_of_assets(self):
        return len(self.data.columns)

    def transaction_cost(self):
        return self._transaction_cost

    def get_data_from_n_days(self, n):
        current_date = self.training_period_start + np.timedelta64(self.current_month, 'M')
        period = current_date.to_period('M')
        return self.data[:period.end_time][-n:].values.T

    @staticmethod
    def __months_diff(earlier, later):
        return 12 * (later.year - earlier.year) + later.month - earlier.month
