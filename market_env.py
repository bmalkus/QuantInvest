import pandas as pd
import numpy as np


class MarketEnv:

    TRANSACTION_COST = 0.019

    def __init__(self):
        self.data = pd.read_csv('./data/Quant_Invest_Fundusze.csv', sep=';', index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        self.data /= self.data.iloc[0, :]

        self.month_to_month = self.data.groupby(pd.Grouper(freq='M'), as_index=False).nth(-1).pct_change() + 1
        self.month_to_month.iloc[0] = self.data.iloc[0]

        self.period_start = None
        self.period_end = None

        # those are indices of months in self.month_to_month
        self.period_start_month = None
        self.current_month = None
        self.period_end_month = None

    def set_period(self, start, end):
        self.period_start = pd.to_datetime(start).replace(day=1)
        self.period_end = pd.to_datetime(end).replace(day=1)

        if self.period_start >= self.period_end:
            raise ValueError('Period start cannot be same as or later than end')

        self.period_start_month = self.__months_diff(self.data.index[0], self.period_start)
        self.period_end_month = self.__months_diff(self.data.index[0], self.period_end)

        self.reset()

    def reset(self):
        self.current_month = self.period_start_month

    def step(self):
        self.current_month += 1

    def progress(self):
        return (self.current_month - self.period_start_month) / (self.period_end_month - self.period_start_month)

    def current_mtm_returns(self):
        return self.month_to_month.iloc[self.current_month].values

    def next_mtm_returns(self):
        return self.month_to_month.iloc[self.current_month + 1].values

    def should_continue(self):
        return self.current_month <= self.period_end_month

    def in_last_month(self):
        return self.current_month == self.period_end_month

    def number_of_assets(self):
        return len(self.data.columns)

    def transaction_cost(self, prev_weights, new_weights):
        return np.sum(np.abs(new_weights[0] - prev_weights[0])) * self.TRANSACTION_COST

    def get_data_from_n_days(self, n, month_offset=0):
        current_date = self.period_start + np.timedelta64(self.current_month + month_offset, 'M')
        period = current_date.to_period('M')
        return self.data[:period.end_time][-n:].values.T

    @staticmethod
    def __months_diff(earlier, later):
        return 12 * (later.year - earlier.year) + later.month - earlier.month
