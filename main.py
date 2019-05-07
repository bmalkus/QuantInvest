import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from PricePredictor import PricePredictor
from market_env import MarketEnv
from utils import results_output_path

np.set_printoptions(edgeitems=10, linewidth=180)


def train_price_predictor(env: MarketEnv, epochs, name, predictor):
    print('Training price predictor:\n'
          '  session name: {}\n'
          '  training data period: {} - {}\n'
          '  number of epochs: {}'.format(name, env.period_start, env.period_end, epochs))

    if predictor.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data')

    env.reset()

    steps = []
    while env.should_continue():
        lookback_prices = env.get_returns_from_n_days(predictor.lookback_window)
        if not env.in_last_month():
            steps.append((lookback_prices, env.next_mtm_returns()))
        env.step()

    predictor.train(steps, epochs)
    predictor.save(name)
    print('Successfully saved session data')


def backtest(env: MarketEnv, name, predictor):
    env.reset()

    if env.current_ind == 0:
        raise ValueError('Backtesting should be offset by at least one month from data start')

    print('Backtesting model:\n'
          '  session name: {}\n'
          '  period: {} - {}'.format(name, env.period_start, env.period_end))

    if predictor.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data, aborting')
        return

    weights = []
    returns = []

    total_reward = 1

    prev_weights = None

    while env.should_continue():
        lookback_prices = env.get_returns_from_n_days(predictor.lookback_window, month_offset=-1)

        predicted = predictor.predict(lookback_prices)
        pred = np.log(1 + predicted)

        if prev_weights is None:
            curr_weights = np.clip(0.2 + pred, 0, 1)
        else:
            curr_weights = prev_weights + np.clip(pred, -0.2, 0.2)
            weights_min = curr_weights.min()
            if weights_min < 0:
                curr_weights += -weights_min
        curr_weights /= curr_weights.sum()

        mtm_return = env.current_mtm_returns()
        r = np.dot(curr_weights, mtm_return)
        if prev_weights is not None:
            r -= env.transaction_cost(prev_weights, curr_weights)

        total_reward *= (1 + r)

        weights.append((env.current_month_timestamp(), *list(curr_weights)))
        returns.append((env.current_month_timestamp(end=True), r))

        prev_weights = curr_weights
        env.step()

    print(total_reward)

    start = weights[0][0]
    end = weights[-1][0]
    date_str = '{}_{}_{}_{}'.format(start.month, start.year, end.month, end.year)

    weights_df = pd.DataFrame(weights, columns=('Date', *env.data.columns))
    weights_df.set_index('Date', inplace=True)
    weights_df.to_csv(os.path.join(results_output_path(name), 'weights_{}.csv'.format(date_str)))

    returns_df = pd.DataFrame(returns, columns=('Date', 'Return'))
    returns_df.set_index('Date', inplace=True)
    returns_df.to_csv(os.path.join(results_output_path(name), 'returns_{}.csv'.format(date_str)))


def main():
    env = MarketEnv()
    env.set_period('2001-01', '2018-12')

    session_name = 'model2'

    predictor = PricePredictor(env, lookback_window=30)

    # train_price_predictor(env, 50, session_name, predictor)
    backtest(env, session_name, predictor)


if __name__ == '__main__':
    main()
