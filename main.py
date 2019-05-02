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

    for epoch in range(1, epochs + 1):
        one_epoch(env, policy)

    policy.save(name)
    print('Successfully saved session data')


def one_epoch(env, policy):
    env.reset()

    steps = []
    total_reward = 1

    init_weights = np.random.rand(1, 7)
    init_weights /= init_weights.sum()

    prev_weights = init_weights
    curr_weights = init_weights

    while env.should_continue():
        # calculate returns for current month and weights
        mtm_returns = env.current_mtm_returns()

        portfolio_change = np.dot(curr_weights, mtm_returns)
        reward = np.log(portfolio_change[0]) - env.transaction_cost(prev_weights, curr_weights)
        total_reward *= (1+reward)

        # predict new weights
        lookback_prices = env.get_data_from_n_days(policy.lookback_window)
        new_weights = policy.predict(lookback_prices, curr_weights)

        # arbitrarily modify new weights to improve exploration
        progress = env.progress()
        if progress <= 1:
            mult = mtm_returns
            predicted = (new_weights * mult.T) / (np.dot(new_weights, mult))
            new_weights = (0.5 + 0.2 * progress) * predicted + (0.5 - 0.2 * progress) * curr_weights
            # new_weights = predicted

        # if env.current_month == 5:
        #     print(curr_weights - predicted)

        if np.random.randint(10) < 1:
            new_weights += (np.random.random(7) - 0.5) / 2
            new_weights = np.abs(new_weights)
            new_weights /= np.sum(new_weights)

        # store current step, later, policy will be trained on that
        if not env.in_last_month():
            future_returns = env.next_mtm_returns()
            steps.append((
                lookback_prices.reshape(lookback_prices.shape[0], lookback_prices.shape[1], 1),
                future_returns,
                curr_weights[0],
                new_weights[0]
            ))

        # go to the next month
        prev_weights, curr_weights = curr_weights, new_weights
        env.step()

        # print(new_weights)
        # print(env.current_mtm_returns())
        # print(reward)

    policy.train(steps)

    print(prev_weights)
    print(total_reward)


def backtest(env: MarketEnv, name, policy):
    env.reset()

    if env.current_month == 0:
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
        if (np.abs(prev_weights - curr_weights) > 0.001).any():
            print(env.current_month - env.period_start_month)
        curr_weights /= np.sum(curr_weights)

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
    # print(weights_df.head(n=50))
    returns_df = pd.DataFrame(returns, columns=['date', 'return'])


def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()


def main():
    # args = parse_args()

    env = MarketEnv()
    env.set_period('2000-02', '2008-10')

    session_name = 'test'

    policy = PolicyPG(env, lookback_window=10)

    train(env, 100, session_name, policy)
    # backtest(env, session_name, policy)


if __name__ == '__main__':
    main()
