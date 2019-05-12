import os

import numpy as np
import pandas as pd

from PricePredictor import PricePredictor
from WeightsPredictor import WeightsPredictor
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


def train_weights_predictor(env: MarketEnv, epochs, name, price_predictor, weights_predictor: WeightsPredictor):
    print('Training weights predictor:\n'
          '  session name: {}\n'
          '  training data period: {} - {}\n'
          '  number of epochs: {}'.format(name, env.period_start, env.period_end, epochs))

    if price_predictor.try_load(name):
        print('Successfully loaded price predictor session data')
    else:
        print('Could not load price predictor session data, aborting')
        return

    if weights_predictor.try_load(name):
        print('Successfully loaded weights predictor session data')
    else:
        print('Could not load weights predictor session data')

    for i in range(epochs):

        print('Epoch # {}'.format(i))

        training_progress = i / epochs

        env.reset()

        prev_weights = None
        total_reward = 1

        steps = []
        while env.should_continue():
            lookback_prices = env.get_returns_from_n_days(price_predictor.lookback_window, month_offset=-1)

            pred_prices = price_predictor.predict(lookback_prices)
            pred_prices = np.clip(pred_prices, a_min=-0.99, a_max=None)
            log_pred_prices = np.log(1 + pred_prices)

            if prev_weights is None:
                curr_weights = np.clip(0.2 + log_pred_prices, 0, 1)
            else:
                calculated_weights = prev_weights + np.clip(log_pred_prices, -0.2, 0.2)
                weights_min = calculated_weights.min()
                if weights_min < 0:
                    calculated_weights += -weights_min
                calculated_weights /= calculated_weights.sum()

                pred_weights = weights_predictor.predict(pred_prices, prev_weights)
                weights_min = pred_weights.min()
                if weights_min < 0:
                    pred_weights += -weights_min
                pred_weights /= pred_weights.sum()

                curr_weights = ((0.8 - 0.3 * training_progress) * calculated_weights +
                                (0.2 + 0.3 * training_progress) * pred_weights)

            curr_weights /= curr_weights.sum()

            mtm_return = env.current_mtm_returns()
            r = np.dot(curr_weights, mtm_return)
            if prev_weights is not None:
                r -= env.transaction_cost(prev_weights, curr_weights)
            total_reward *= (1 + r)

            if prev_weights is not None:
                steps.append((pred_prices, prev_weights, curr_weights))

            prev_weights = curr_weights
            env.step()

        print(total_reward)
        print(curr_weights)

        weights_predictor.train(steps, epochs=3, verbose=0)

    weights_predictor.save(name)
    print('Successfully saved session data')


def backtest(env: MarketEnv, name, price_predictor, weights_predictor: WeightsPredictor = None):
    env.reset()

    if env.current_ind == 0:
        raise ValueError('Backtesting should be offset by at least one month from data start')

    print('Backtesting model:\n'
          '  session name: {}\n'
          '  period: {} - {}'.format(name, env.period_start, env.period_end))

    if price_predictor.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data, aborting')
        return

    if weights_predictor is not None:
        if weights_predictor.try_load(name):
            print('Successfully loaded weights predictor session data')
        else:
            print('Could not load weights predictor session data, aborting')
            return

    weights = []
    returns = []

    total_reward = 1

    prev_weights = None

    while env.should_continue():
        lookback_prices = env.get_returns_from_n_days(price_predictor.lookback_window, month_offset=-1)

        prices_pred = price_predictor.predict(lookback_prices)
        log_prices_pred = np.log(1 + prices_pred)

        if prev_weights is None:
            curr_weights = np.clip(0.2 + log_prices_pred, 0, 1)
        elif weights_predictor is not None:
            curr_weights = weights_predictor.predict(prices_pred, prev_weights)
            weights_min = curr_weights.min()
            if weights_min < 0:
                curr_weights += -weights_min
        else:
            curr_weights = prev_weights + np.clip(log_prices_pred, -0.2, 0.2)
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
    env.set_period('2001-01', '2015-12')

    session_name = 'model3'

    weights_predictor = WeightsPredictor(env)
    price_predictor = PricePredictor(env, lookback_window=30)

    # train_price_predictor(env, 250, session_name, price_predictor)
    # train_weights_predictor(env, 50, session_name, price_predictor, weights_predictor)
    backtest(env, session_name, price_predictor, weights_predictor)
    # backtest(env, session_name, price_predictor)


if __name__ == '__main__':
    main()
