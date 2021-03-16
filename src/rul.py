import warnings
import uuid

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from utilities import *
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
from ECLSTM import ECLSTM1D
from ECLSTM import check_the_config_valid
from ECLSTM import build_the_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from parameters import architecture_parameters


class RemainingUsefulLife:
    """Class which predicts the remaining useful life.

    Attributes:
        data: Data to predict the remaining useful life for.
        data_id: ID of the given dataframe.
        sequence_length:
        window_size:
        model_id: Generated uuid.
        train_FD:
        test_FD:
        X_batch:
        y_batch:
        number_of_sensors:
        log_dir:
    """
    def __init__(self, data: pd.DataFrame, data_id: str = '', sequence_length: int = 5, window_size: int = 16) -> None:
        """Inits RemainingUsefulLife."""
        self.__data = data
        self.__data_id = data_id
        self.__model = None
        self.__model_id = uuid.uuid1()
        self.__sequence_length = sequence_length
        self.__window_size = window_size
        self.__train_FD = None
        self.__test_FD = None
        self.__X_batch = None
        self.__y_batch = None
        self.__number_of_sensors = None
        self.__log_dir = None

    def rul(self, max_life: int) -> None:
        id = 'engine_id'
        rul = []
        for _id in set(self.__train_FD[id]):
            trainFD_of_one_id = self.__train_FD[self.__train_FD[id] == _id]
            cycle_list = trainFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list)

            knee_point = max_cycle - max_life
            kink_rul = []

            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_rul.append(max_life)
                else:
                    tmp = max_cycle - i - 1
                    kink_rul.append(tmp)
            rul.extend(kink_rul)

        self.__train_FD["RUL"] = rul

        id = 'engine_id'
        rul = []
        for _id_test in set(self.__test_FD[id]):
            true_rul = int(RUL_FD.iloc[_id_test - 1])
            testFD_of_one_id = self.__test_FD[self.__test_FD[id] == _id_test]
            cycle_list = testFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list) + true_rul
            knee_point = max_cycle - max_life
            kink_rul = []

            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_rul.append(max_life)
                else:
                    tmp = max_cycle - i - 1
                    kink_rul.append(tmp)

            rul.extend(kink_rul)

        self.__test_FD["RUL"] = rul

    def data_prep(self) -> None:
        col_to_drop = identify_and_remove_unique_columns(self.__train_FD)
        self.__train_FD = self.__train_FD.drop(col_to_drop, axis=1)
        self.__test_FD = self.__test_FD.drop(col_to_drop, axis=1)
        mean = self.__train_FD.iloc[:, 2:-1].mean()
        std = self.__train_FD.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)

        # training dataset
        self.__train_FD.iloc[:, 2:-1] = (self.__train_FD.iloc[:, 2:-1] - mean) / std

        # Testing dataset
        self.__test_FD.iloc[:, 2:-1] = (self.__test_FD.iloc[:, 2:-1] - mean) / std
        training_data = self.__train_FD.values
        testing_data = self.__test_FD.values

        x_train = training_data[:, 2:-1]
        y_train = training_data[:, -1]
        print(f'Training: {x_train.shape}, {y_train.shape}')

        x_test = testing_data[:, 2:-1]
        y_test = testing_data[:, -1]
        print(f'Testing: {x_test.shape}, {y_test.shape}')

        # Prepare the training set according to the  window size and sequence_length
        self.__X_batch, self.__y_batch = batch_generator(self.__train_FD,
                                                         sequence_length=self.__sequence_length,
                                                         window_size=self.__window_size)

        self.__X_batch = np.expand_dims(self.__X_batch, axis=4)
        self.__y_batch = np.expand_dims(self.__y_batch, axis=1)
        self.__number_of_sensors = self.__X_batch.shape[-2]

    def initialize_model(self) -> None:
        valid = check_the_config_valid(architecture_parameters, self.__window_size, self.__number_of_sensors)
        if valid:
            self.__model = build_the_model(architecture_parameters, self.__sequence_length, self.__window_size,
                                           self.__number_of_sensors)
        else:
            print("invalid configuration")

        input_data = tf.keras.layers.Input(shape=(self.__sequence_length, self.__window_size, self.__number_of_sensors,
                                                  1))
        out = self.__model(input_data)
        print(self.__model.summary())

    def train_model(self) -> None:
        date_time_obj = datetime.now()
        self.__log_dir = "logs/{}_{}_{}_{}_{}_{}_{}/".format(self.__data_id, date_time_obj.year, date_time_obj.month,
                                                             date_time_obj.day, date_time_obj.hour,
                                                             date_time_obj.minute, date_time_obj.second)

        os.makedirs(self.__log_dir)

        checkpoint = ModelCheckpoint(self.__log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
        self.__model.fit(self.__X_batch,
                         self.__y_batch,
                         batch_size=15,
                         epochs=30,
                         callbacks=[checkpoint, reduce_lr, early_stopping],
                         validation_split=0.075)

        self.__model.save_weights(self.__log_dir + 'trained_weights_final.h5')

    def evaluate_model(self) -> None:
        model_list = os.listdir(self.__log_dir)
        model_list = [file for file in model_list if "val_loss" in file]
        self.__model.load_weights(self.__log_dir + model_list[-1])

        y_batch_pred = self.__model.predict(self.__X_batch)
        y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])
        y_batch_reshape = self.__y_batch.reshape(self.__y_batch.shape[0], self.__y_batch.shape[1])
        rmse_on_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))

        print(f'The RMSE on Training dataset {self.__data_id} is {rmse_on_train}.')

        x_batch_test, y_batch_test = test_batch_generator(self.__test_FD,
                                                          sequence_length=self.__sequence_length,
                                                          window_size=self.__window_size)

        x_batch_test = np.expand_dims(x_batch_test, axis=4)
        model_weights = f'trained_models/{self.__data_id}/best_model_{self.__data_id}_{self.__model_id}.h5'
        if not os.path.exists(model_weights):
            print("no such a weight file")
        else:
            self.__model.load_weights(model_weights)

        y_batch_pred_test = self.__model.predict(x_batch_test)
        rmse_on_test = np.sqrt(mean_squared_error(y_batch_pred_test, y_batch_test))

        print(f'The RMSE on test dataset {self.__data_id} is {rmse_on_test}.')

    def get_model(self) -> object:
        """Returns the model."""
        return self.__model
