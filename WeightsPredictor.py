import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OpError
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Reshape, Concatenate, Conv2D, Flatten, Dense
from tensorflow.python.keras.models import load_model

from market_env import MarketEnv
from utils import results_output_path

tf.logging.set_verbosity(tf.logging.ERROR)


class WeightsPredictor:

    def __init__(self, env: MarketEnv):
        self.learning_rate = 10e-3

        self.env = env
        self.number_of_assets = self.env.number_of_assets()

        self.model = None
        self.price_pred = None
        self.prev_weights = None

        self.__build_model()

    def __get_loss_function(self):

        def loss_function(_, weights_pred):
            return K.math_ops.neg(K.sum(K.math_ops.multiply(weights_pred, self.price_pred), axis=1) - self.tc(weights_pred))
            # return K.math_ops.neg(K.log(K.sum(K.math_ops.multiply(weights_pred, self.price_pred), axis=1) - self.tc(weights_pred)))

        return loss_function

    def tc(self, new_weights):
        return K.sum(K.abs(new_weights - self.prev_weights), axis=1) * self.env.TRANSACTION_COST

    def __build_model(self):

        self.price_pred = Input((self.number_of_assets, ))
        self.prev_weights = Input((self.number_of_assets, ))

        price_pred_reshaped = Reshape((self.number_of_assets, 1, 1))(self.price_pred)
        prev_weights_reshaped = Reshape((self.number_of_assets, 1, 1))(self.prev_weights)

        network = Concatenate(axis=2)([price_pred_reshaped, prev_weights_reshaped])

        network = Conv2D(16, (1, 2), activation='relu', kernel_initializer='VarianceScaling')(network)

        network = Flatten()(network)

        network = Dense(network.get_shape()[1], 'relu', kernel_initializer='VarianceScaling')(network)

        new_weights = Dense(self.number_of_assets, 'softmax', kernel_initializer='VarianceScaling')(network)

        self.model = tf.keras.models.Model(inputs=[self.price_pred, self.prev_weights], outputs=new_weights)

        self.model.compile(optimizer='adam', loss=self.__get_loss_function())

    def train(self, steps, epochs, verbose=1):
        price_predictions = np.array([step[0] for step in steps])
        prev_weights = np.array([step[1] for step in steps])
        new_weights = np.array([step[2] for step in steps])
        self.model.fit(x=[price_predictions, prev_weights], y=new_weights, epochs=epochs, verbose=verbose)

    def predict(self, price_pred, prev_weights):
        return self.model.predict([price_pred.reshape(1, price_pred.shape[0]),
                                   prev_weights.reshape(1, prev_weights.shape[0])])[0]

    @staticmethod
    def __network_output_path(name):
        return os.path.join(results_output_path(name), 'wp', '{}'.format(name))

    def try_load(self, name):
        try:
            # self.model = load_model(self.__network_output_path(name), custom_objects={'loss_function': self.__get_loss_function()})
            self.model.load_weights(self.__network_output_path(name))
            return True
        except (OpError, OSError):
            return False

    def save(self, name):
        path = self.__network_output_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # self.model.save(path)
        self.model.save_weights(path)
