import os

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError

tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import tflearn

from market_env import MarketEnv


class PolicyPG:

    def __init__(self, env: MarketEnv, lookback_window):
        self.learning_rate = 10e-3

        self.env = env
        self.lookback_window = lookback_window
        self.number_of_assets = self.env.number_of_assets()

        self.prices_lookback = None
        self.prev_weights = None
        self.new_weights = None

        self.__build_network()

        self.session = tf.Session()

        self.future_returns = tf.placeholder(tf.float32, [None, self.number_of_assets])
        self.pv_vector = tf.reduce_sum(tf.multiply(self.new_weights, self.future_returns), reduction_indices=[1])
        self.profit = tf.reduce_prod(self.pv_vector) - self.tc()
        self.loss = -tf.reduce_mean(tf.log(self.pv_vector))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def tc(self):
        return tf.reduce_sum(tf.abs(self.new_weights - self.prev_weights), axis=1) * 0.019

    def __build_network(self):
        prices = tf.placeholder(
            tf.float32,
            shape=[None, self.number_of_assets, self.lookback_window, 1],
            name='asset_prices'
        )
        network = tflearn.layers.conv_2d(
            prices,
            2,
            [1, 4],
            [1, 1, 1, 1],
            'valid',
            'relu'
        )
        network = tflearn.layers.conv_2d(
            network,
            # 48,
            1,
            [1, network.get_shape()[2]],
            [1, 1],
            'valid',
            'relu',
            regularizer='L2',
            # weight_decay=5e-9
        )

        w_previous = tf.placeholder(tf.float32, shape=[None, self.number_of_assets], name='previous_weights')

        network = tf.concat([network, tf.reshape(w_previous, [-1, self.number_of_assets, 1, 1])], axis=2)
        network = tflearn.layers.conv_2d(
            network,
            1,
            [1, network.get_shape()[2]],
            [1, 1],
            'valid',
            'relu',
            regularizer='L2',
            # weight_decay=5e-9
        )
        network = tf.layers.flatten(network)
        w_init = tf.random_uniform_initializer(-1, 1)
        out = tf.layers.dense(network, self.number_of_assets, activation=tf.nn.softmax, kernel_initializer=w_init,
                              name='new_weights')

        self.prices_lookback = prices
        self.prev_weights = w_previous
        self.new_weights = out

    def train(self, steps):
        prices_lookback = [step[0] for step in steps]
        future_returns = [step[1] for step in steps]
        prev_weights = [step[2] for step in steps]
        new_weights = [step[3] for step in steps]
        profit, _ = self.session.run(
            [self.profit, self.optimize],
            feed_dict={
                self.prices_lookback: np.array(prices_lookback),
                self.future_returns: np.array(future_returns),
                self.new_weights: np.reshape(new_weights, (-1, self.number_of_assets)),
                self.prev_weights: np.reshape(prev_weights, (-1, self.number_of_assets))
            }
        )

    def predict(self, prices_history, curr_weights):
        prices_history = prices_history.reshape(1, prices_history.shape[0], prices_history.shape[1], 1)
        return self.session.run(
            self.new_weights,
            feed_dict={
                self.prices_lookback: prices_history,
                self.prev_weights: curr_weights
            }
        )

    @staticmethod
    def __build_path(name):
        return './results/{0}/{0}'.format(name)

    def try_load(self, name):
        try:
            self.saver.restore(self.session, self.__build_path(name))
            return True
        except (ValueError, NotFoundError):
            return False

    def save(self, name):
        path = self.__build_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.saver.save(self.session, path)
