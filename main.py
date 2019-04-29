from argparse import ArgumentParser

import numpy as np

from PolicyPG import PolicyPG
from market_env import MarketEnv


def train(env, epochs):
    print('Training model:\n'
          '  training data period: {} - {}\n'
          '  number of epochs: {}'.format(env.training_period_start, env.training_period_end, epochs))

    policy = PolicyPG(env, 10)

    for epoch in range(1, epochs + 1):
        one_epoch(env, policy)


def one_epoch(env, policy):
    env.reset()

    steps = []
    total_reward = 0

    init_weights = np.random.rand(1, 7)
    init_weights /= init_weights.sum()

    prev_weights = init_weights
    while env.should_continue():
        lookback_prices = env.get_data_from_n_days(policy.lookback_window)
        new_weights = policy.predict(lookback_prices, prev_weights)

        env.step()

        mtm_returns = env.current_mtm_returns()

        progress = env.progress()
        if progress < 0.45:
            predicted = (new_weights * mtm_returns.T) / (np.dot(new_weights, mtm_returns))
            random = np.random.rand(1, 7)
            random /= random.sum()
            new_weights = (0.1 + 2 * progress) * predicted + (0.9 - 2 * progress) * init_weights

        portfolio_change = np.dot(new_weights, mtm_returns)
        reward = np.log(portfolio_change[0])
        total_reward += reward

        future_returns = env.next_mtm_returns()

        if env.should_continue():
            steps.append((
                lookback_prices.reshape(lookback_prices.shape[0], lookback_prices.shape[1], 1),
                future_returns,
                prev_weights[0],
                new_weights[0]
            ))

        prev_weights = new_weights

        # print(new_weights)
        # print(env.current_mtm_returns())
        # print(reward)

    policy.train(steps)

    print(new_weights)
    print(total_reward)


def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()


def main():
    # args = parse_args()

    env = MarketEnv()
    env.set_training_period('2000-01-30', '2010-12-02')

    train(env, 100)


if __name__ == '__main__':
    main()
