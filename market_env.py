import numpy as np
import pandas as pd


class MarketEnv:

    TRANSACTION_COST = 0.019

    def __init__(self):
        self.data = pd.read_csv('./data/Quant_Invest_Fundusze.csv', sep=';', index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        self.data /= self.data.iloc[0, :]

        self.month_to_month = self.data.groupby(pd.Grouper(freq='M'), as_index=False).nth(-1).pct_change() + 1
        self.month_to_month.iloc[0] = self.data.iloc[0]

        self.day_to_day = self.data.pct_change() + 1
        self.day_to_day.iloc[0] = self.data.iloc[0]

        self.period_start = None
        self.period_end = None

        # those are indices of months in self.month_to_month
        self.period_start_ind = None
        self.current_ind = None
        self.period_end_ind = None

    def set_period(self, start, end):
        self.period_start = pd.to_datetime(start).replace(day=1)
        self.period_end = pd.to_datetime(end).replace(day=1)

        if self.period_start >= self.period_end:
            raise ValueError('Period start cannot be same as or later than end')

        self.period_start_ind = self.__months_diff(self.data.index[0], self.period_start)
        self.period_end_ind = self.__months_diff(self.data.index[0], self.period_end)
        self.reset()

    def reset(self):
        self.current_ind = self.period_start_ind

    def step(self):
        self.current_ind += 1

    def progress(self):
        return (self.current_ind - self.period_start_ind) / (self.period_end_ind - self.period_start_ind)

    def current_month_timestamp(self, end=False):
        date = self.data.index[0] + np.timedelta64(self.current_ind, 'M')
        period = date.to_period('M')
        if end:
            date = period.end_time
        else:
            date = period.start_time
        return date

    def current_mtm_returns(self):
        return self.month_to_month.iloc[self.current_ind].values - 1

    def next_mtm_returns(self):
        return self.month_to_month.iloc[self.current_ind + 1].values - 1

    def should_continue(self):
        return self.current_ind <= self.period_end_ind

    def in_last_month(self):
        return self.current_ind == self.period_end_ind

    def number_of_assets(self):
        return len(self.data.columns)

    def transaction_cost(self, prev_weights, new_weights):
        return np.sum(np.abs(new_weights - prev_weights)) * self.TRANSACTION_COST

    def get_returns_from_n_days(self, n, month_offset=0):
        current_date = self.data.index[0] + np.timedelta64(self.current_ind + month_offset, 'M')
        period = current_date.to_period('M')
        ret = self.data[:period.end_time][-n:].values
        ret /= ret[0]
        return ret.T - 1

    @staticmethod
    def __months_diff(earlier, later):
        return 12 * (later.year - earlier.year) + later.month - earlier.month
