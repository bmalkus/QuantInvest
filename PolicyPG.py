import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OpError

from market_env import MarketEnv

tf.logging.set_verbosity(tf.logging.ERROR)


class PolicyPG:

    model: tf.keras.models.Sequential

    def __init__(self, env: MarketEnv, lookback_window):
        self.learning_rate = 10e-3
        # self.learning_rate = 0.05

        self.env = env
        self.lookback_window = lookback_window
        self.number_of_assets = self.env.number_of_assets()

        self.prices_lookback = None
        self.prev_weights = None
        self.new_weights = None

        self.model = None

        self.__build_network()

        # self.session = tf.Session()

        # self.future_returns = tf.placeholder(tf.float32, [None, self.number_of_assets])
        # self.pv_vector = tf.reduce_sum(tf.multiply(self.new_weights, self.future_returns), reduction_indices=[1])
        # self.profit = tf.reduce_prod(self.pv_vector) - self.tc()
        # self.loss = -tf.reduce_mean(tf.log(self.pv_vector))
        # self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #
        # self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def tc(self):
        return tf.reduce_sum(tf.abs(self.new_weights - self.prev_weights), axis=1) * 0.019

    def __build_network(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.InputLayer((self.number_of_assets, self.lookback_window, 1)))

        model.add(tf.keras.layers.Conv2D(32, (1, 10), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))

        model.add(tf.keras.layers.Conv2D(128, (1, 5), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))

        model.add(tf.keras.layers.Conv2D(256, (1, 5), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))
        #
        # model.add(tf.keras.layers.Conv1D(128, 2, activation='relu', kernel_initializer='VarianceScaling'))
        #
        # model.add(tf.keras.layers.AveragePooling1D(2, 1))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(model.layers[-1].output_shape[1], 'relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.Dense(self.number_of_assets, 'linear', kernel_initializer='VarianceScaling'))

        model.compile(optimizer='adam', loss='mse')

        self.model = model

        # w_previous = tf.placeholder(tf.float32, shape=[None, self.number_of_assets], name='previous_weights')
        #
        # prices = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self.number_of_assets, self.lookback_window, 1],
        #     name='asset_prices'
        # )
        # network = tflearn.layers.conv_2d(
        #     prices,
        #     8,
        #     [1, 3],
        #     [1, 1, 1, 1],
        #     'valid',
        #     'relu'
        # )
        # network = tflearn.layers.avg_pool_2d(
        #     network,
        #     [1, 3],
        #     1,
        #     'valid'
        # )
        # network = tflearn.layers.conv_2d(
        #     network,
        #     64,
        #     [1, 3],
        #     [1, 1],
        #     'valid',
        #     'relu',
        #     regularizer='L2',
        #     # weight_decay=5e-9
        # )
        # network = tflearn.layers.avg_pool_2d(
        #     network,
        #     [1, 3],
        #     1,
        #     'valid'
        # )
        # network = tflearn.layers.conv_2d(
        #     network,
        #     256,
        #     [1, 2],
        #     [1, 1],
        #     'valid',
        #     'relu',
        #     regularizer='L2',
        #     # weight_decay=5e-9
        # )
        #
        # network = tf.layers.flatten(network)
        # network = tf.layers.dense(network, network.get_shape()[1], activation=tf.nn.relu)
        # # network = tf.concat([network, tf.reshape(w_previous, [-1, self.number_of_assets])], axis=1)
        #
        # w_init = tf.random_uniform_initializer(-0.05, 0.05)
        # out = tf.layers.dense(network, self.number_of_assets, activation=tf.nn.softmax, kernel_initializer=w_init)
        #
        # self.prices_lookback = prices
        # self.prev_weights = w_previous
        # self.new_weights = out

    def train(self, steps, epochs):
        prices_lookback = np.array([step[0] for step in steps])
        future_returns = np.array([step[1] for step in steps])
        # print(prices_lookback.shape)
        self.model.fit(x=prices_lookback, y=future_returns, epochs=epochs)
        # prev_weights = [step[2] for step in steps]
        # new_weights = [step[3] for step in steps]
        # profit, _ = self.session.run(
        #     [self.profit, self.optimize],
        #     feed_dict={
        #         self.prices_lookback: np.array(prices_lookback),
        #         self.future_returns: np.array(future_returns),
        #         self.new_weights: np.reshape(new_weights, (-1, self.number_of_assets)),
        #         self.prev_weights: np.reshape(prev_weights, (-1, self.number_of_assets))
        #     }
        # )

    def predict(self, prices_history, curr_weights):
        p = np.array([prices_history])
        return self.model.predict(p)[0]
        # prices_history = prices_history.reshape(1, prices_history.shape[0], prices_history.shape[1], 1)
        # return self.session.run(
        #     self.new_weights,
        #     feed_dict={
        #         self.prices_lookback: prices_history,
        #         self.prev_weights: curr_weights
        #     }
        # )

    @staticmethod
    def __network_output_path(name):
        return os.path.join(PolicyPG.output_path(name), name)

    @staticmethod
    def output_path(name):
        return os.path.join('.', 'results', name)

    def try_load(self, name):
        try:
            self.model.load_weights(self.__network_output_path(name))
            return True
        except OpError:
            return False

    def save(self, name):
        path = self.__network_output_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)
