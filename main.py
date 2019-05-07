import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from PolicyPG import PolicyPG
from market_env import MarketEnv

np.set_printoptions(edgeitems=10, linewidth=180)


def train(env: MarketEnv, epochs, name, policy):
    print('Training model:\n'
          '  session name: {}\n'
          '  training data period: {} - {}\n'
          '  number of epochs: {}'.format(name, env.period_start, env.period_end, epochs))

    if policy.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data')

    # for epoch in range(1, epochs + 1):
    one_epoch(env, policy, epochs)

    policy.save(name)
    print('Successfully saved session data')


def one_epoch(env, policy, epochs):
    env.reset()

    steps = []
    total_reward = 1

    init_weights = np.random.rand(1, 7)
    init_weights /= init_weights.sum()

    prev_weights = init_weights
    curr_weights = init_weights

    while env.should_continue():
        # calculate returns for current month and weights
        # mtm_returns = env.current_mtm_returns()
        #
        # portfolio_change = np.dot(curr_weights, mtm_returns)
        # reward = np.log(portfolio_change[0]) - env.transaction_cost(prev_weights, curr_weights)
        # total_reward *= (1+reward)
        #
        # # predict new weights
        lookback_prices = env.get_data_from_n_days(policy.lookback_window)
        # predicted = policy.predict(lookback_prices, curr_weights)
        # predicted = (predicted * mtm_returns.T) / (np.dot(predicted, mtm_returns))
        #
        # # arbitrarily modify new weights to improve exploration
        #
        # curr_modified = curr_weights
        # if np.random.randint(10) < 1:
        #     curr_modified += (np.random.random(7) - 0.5) / 5
        #     curr_modified = np.abs(curr_modified)
        #     curr_modified /= np.sum(curr_modified)
        #
        # new_weights = 0.1 * predicted + 0.9 * curr_modified

        # new_weights = (curr_weights * mult.T) / np.dot(curr_weights, mult)
        # if env.current_month == 5:
        #     print(curr_weights - predicted)

        # store current step, later, policy will be trained on that
        if not env.in_last_month():
            future_returns = env.next_mtm_returns()
            steps.append((
                lookback_prices.reshape(lookback_prices.shape[0], lookback_prices.shape[1], 1),
                future_returns
            ))
        # break

        # go to the next month
        # prev_weights, curr_weights = curr_weights, new_weights
        env.step()

        # print(new_weights)
        # print(env.current_mtm_returns())
        # print(reward)

    # print(steps)

    policy.train(steps, epochs)

    # print(prev_weights)
    # print(total_reward)


def backtest(env: MarketEnv, name, policy):
    env.reset()

    if env.current_ind == 0:
        raise ValueError('Backtesting should be offset by at least one month from data start')

    print('Backtesting model:\n'
          '  session name: {}\n'
          '  period: {} - {}'.format(name, env.period_start, env.period_end))

    if policy.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data, aborting')
        return

    total_reward = 1
    weights = []
    returns = []

    prev_weights = np.array([[0, 0, 0, 0, 0, 0, 0]])

    while env.should_continue():
        lookback_prices = env.get_data_from_n_days(policy.lookback_window, month_offset=-1)

        curr_weights = policy.predict(lookback_prices, prev_weights)
        curr_weights /= np.sum(curr_weights)

        # if (np.abs(prev_weights - curr_weights) > 0.001).any():
        #     print(env.current_month - env.period_start_month)
        # if (prev_weights != curr_weights).any():
        #     print(np.abs(prev_weights - curr_weights))

        mtm_returns = env.current_mtm_returns()

        portfolio_change = np.dot(curr_weights, mtm_returns)
        ret = portfolio_change[0]
        if np.sum(prev_weights) != 0:
            ret -= env.transaction_cost(prev_weights, curr_weights)

        total_reward *= ret

        weights.append((env.current_month_timestamp(), *list(curr_weights[0])))
        returns.append((env.current_month_timestamp(), ret))

        prev_weights = curr_weights
        env.step()

    print(total_reward)
    weights_df = pd.DataFrame(weights, columns=['date', *env.data.columns])
    # print(weights_df.head(n=10))
    returns_df = pd.DataFrame(returns, columns=['date', 'return'])


def backtest2(env: MarketEnv, name, policy):
    env.reset()

    if env.current_ind == 0:
        raise ValueError('Backtesting should be offset by at least one month from data start')

    print('Backtesting model:\n'
          '  session name: {}\n'
          '  period: {} - {}'.format(name, env.period_start, env.period_end))

    if policy.try_load(name):
        print('Successfully loaded session data')
    else:
        print('Could not load session data, aborting')
        return

    weights = []
    returns = []

    total_reward = 1

    prev_weights = None

    while env.should_continue():
        lookback_prices = env.get_data_from_n_days(policy.lookback_window, month_offset=-1)

        # print(lookback_prices)
        predicted = policy.predict(lookback_prices.reshape(lookback_prices.shape[0], lookback_prices.shape[1], 1), prev_weights) / 10
        pred = np.log(1 + predicted)

        # if (np.abs(prev_weights - curr_weights) > 0.001).any():
        #     print(env.current_month - env.period_start_month)
        # if (prev_weights != curr_weights).any():
        #     print(np.abs(prev_weights - curr_weights))

        if prev_weights is None:
            curr_weights = np.clip(0.2 + pred, 0, 1)
        else:
            curr_weights = prev_weights + np.clip(pred, -0.2, 0.2)
            weights_min = curr_weights.min()
            if weights_min < 0:
                curr_weights += -weights_min
        curr_weights /= curr_weights.sum()
        mtm_return = env.current_mtm_returns() / 10
        # print(curr_weights)
        # print(pred)
        # print(mtm_return)
        r = np.dot(curr_weights, mtm_return)
        if prev_weights is not None:
            r -= env.transaction_cost(prev_weights, curr_weights)
        # print(r)
        # print('-------')
        total_reward *= (1 + r)
        # print('act:  {}'.format(mtm_return))
        # print('diff: {}'.format((mtm_return - pred)))

        # portfolio_change = np.dot(curr_weights, mtm_returns)
        # ret = portfolio_change[0]
        # if np.sum(prev_weights) != 0:
        #     ret -= env.transaction_cost(prev_weights, curr_weights)

        # total_reward *= ret

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
    weights_df.to_csv(os.path.join(policy.output_path(name), 'weights_{}.csv'.format(date_str)))

    returns_df = pd.DataFrame(returns, columns=('Date', 'Return'))
    returns_df.set_index('Date', inplace=True)
    returns_df.to_csv(os.path.join(policy.output_path(name), 'returns_{}.csv'.format(date_str)))


def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()


def main():
    # args = parse_args()

    env = MarketEnv()
    env.set_period('2001-01', '2018-12')

    session_name = 'model1'

    policy = PolicyPG(env, lookback_window=30)

    # train(env, 50, session_name, policy)
    backtest2(env, session_name, policy)


if __name__ == '__main__':
    main()
