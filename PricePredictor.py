import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OpError

from market_env import MarketEnv
from utils import results_output_path

tf.logging.set_verbosity(tf.logging.ERROR)


class PricePredictor:

    def __init__(self, env: MarketEnv, lookback_window):
        self.learning_rate = 10e-3

        self.env = env
        self.lookback_window = lookback_window
        self.number_of_assets = self.env.number_of_assets()

        self.prices_lookback = None
        self.prev_weights = None
        self.new_weights = None

        self.model = None

        self.__build_network()

    def __build_network(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.InputLayer((self.number_of_assets, self.lookback_window, 1)))

        model.add(tf.keras.layers.Conv2D(32, (1, 10), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))

        model.add(tf.keras.layers.Conv2D(128, (1, 5), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))

        model.add(tf.keras.layers.Conv2D(256, (1, 5), activation='relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.MaxPool2D((1, 5), 1))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(model.layers[-1].output_shape[1], 'relu', kernel_initializer='VarianceScaling'))

        model.add(tf.keras.layers.Dense(self.number_of_assets, 'linear', kernel_initializer='VarianceScaling'))

        model.compile(optimizer='adam', loss='mse')

        self.model = model

    def train(self, steps, epochs):
        prices_lookback = np.array([step[0].reshape(step[0].shape[0], step[0].shape[1], 1) for step in steps])
        future_returns = np.array([step[1] for step in steps])
        self.model.fit(x=prices_lookback * 10, y=future_returns * 10, epochs=epochs)

    def predict(self, prices_history):
        p = np.array([prices_history.reshape(prices_history.shape[0], prices_history.shape[1], 1)])
        return self.model.predict(p * 10)[0] / 10

    @staticmethod
    def __network_output_path(name):
        return os.path.join(results_output_path(name), name)

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
