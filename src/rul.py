import warnings
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


class RemainingUsefulLife:
    """Class which predicts the remaining useful life.

    Attributes:
        data: Data to predict the remaining useful life for.
        data_id: 
    """
    def __init__(self, data: pd.DataFrame, data_id: str = '') -> None:
        """Inits RemainingUsefulLife."""
        self.__data = data
        self.__data_id = data_id
        self.__model = None
        self.__model_id = None

    def rul(self, max_life: int) -> None:
        id = 'engine_id'
        MAXLIFE = 120  # or 125 , 130
        # piecewise linear RUL
        rul = []
        for _id in set(train_FD[id]):
            trainFD_of_one_id = train_FD[train_FD[id] == _id]
            cycle_list = trainFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list)

            knee_point = max_cycle - MAXLIFE
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                #
                if i < knee_point:
                    kink_RUL.append(MAXLIFE)
                else:
                    tmp = max_cycle - i - 1
                    kink_RUL.append(tmp)
            rul.extend(kink_RUL)

        train_FD["RUL"] = rul

        id = 'engine_id'
        rul = []
        for _id_test in set(test_FD[id]):
            true_rul = int(RUL_FD.iloc[_id_test - 1])
            testFD_of_one_id = test_FD[test_FD[id] == _id_test]
            cycle_list = testFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list) + true_rul
            knee_point = max_cycle - MAXLIFE
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_RUL.append(MAXLIFE)
                else:
                    tmp = max_cycle - i - 1
                    kink_RUL.append(tmp)

            rul.extend(kink_RUL)

        test_FD["RUL"] = rul

    def data_prep(self) -> None:
        col_to_drop = identify_and_remove_unique_columns(train_FD)
        train_FD = train_FD.drop(col_to_drop, axis=1)
        test_FD = test_FD.drop(col_to_drop, axis=1)
        mean = train_FD.iloc[:, 2:-1].mean()
        std = train_FD.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)

        # training dataset
        train_FD.iloc[:, 2:-1] = (train_FD.iloc[:, 2:-1] - mean) / std

        # Testing dataset
        test_FD.iloc[:, 2:-1] = (test_FD.iloc[:, 2:-1] - mean) / std
        training_data = train_FD.values
        testing_data = test_FD.values

        x_train = training_data[:, 2:-1]
        y_train = training_data[:, -1]
        print("training", x_train.shape, y_train.shape)

        x_test = testing_data[:, 2:-1]
        y_test = testing_data[:, -1]
        print("testing", x_test.shape, y_test.shape)
        sequence_length = 5
        window_size = 16
        # Prepare the training set according to the  window size and sequence_length
        x_batch, y_batch = batch_generator(train_FD, sequence_length=sequence_length, window_size=window_size)

        x_batch = np.expand_dims(x_batch, axis=4)
        y_batch = np.expand_dims(y_batch, axis=1)
        number_of_sensor = x_batch.shape[-2]

    def initialize_model(self) -> None:
        para = {
            # preprocessing part
            "preprocessing_layers": 0,
            "pre_kernel_width": 3,
            "pre_number_filters": 10,
            "pre_strides": 2,
            "pre_activation": "relu",

            # ECLSTM feature extraction part
            "eclstm_1_recurrent_activation": ['linear', "hard_sigmoid"],
            "eclstm_1_conv_activation": ['hard_sigmoid', "hard_sigmoid"],
            "eclstm_1_kernel_width": [3, 3],
            "eclstm_1_number_filters": [10, 10],
            "eclstm_1_strides": 1,
            "eclstm_1_fusion": ["early", "early"],
            "eclstm_2_recurrent_activation": ['linear', "hard_sigmoid"],
            "eclstm_2_conv_activation": ['hard_sigmoid', "hard_sigmoid"],
            "eclstm_2_kernel_width": [3, 3],
            "eclstm_2_number_filters": [20, 20],
            "eclstm_2_strides": 1,
            "eclstm_2_fusion": ["early", "early"],
            "eclstm_3_recurrent_activation": [None],
            "eclstm_3_conv_activation": [None],
            "eclstm_3_kernel_width": [None],
            "eclstm_3_number_filters": [None],
            "eclstm_3_strides": None,
            "eclstm_3_fusion": [None],
            "eclstm_4_recurrent_activation": [None],
            "eclstm_4_conv_activation": [None],
            "eclstm_4_kernel_width": [None],
            "eclstm_4_number_filters": [None],
            "eclstm_4_strides": None,
            "eclstm_4_fusion": [None],

            # Prediction
            "prediction_1_filters": 150,
            "prediction_1_activation": "relu",
            "prediction_2_filters": 0,
            "prediction_2_activation": None,
            "prediction_3_filters": 0,
            "prediction_3_activation": None,
            "prediction_4_filters": 0,
            "prediction_4_activation": None,
        }
        valid = check_the_config_valid(para, window_size, number_of_sensor)
        if valid:
            model = build_the_model(para, sequence_length, window_size, number_of_sensor)
        else:
            print("invalid configuration")
        input_data = tf.keras.layers.Input(shape=(sequence_length, window_size, number_of_sensor, 1))
        out = model(input_data)
        print(model.summary())

    def train_model(self, data_id, model, x_batch, y_batch) -> None:
        dateTimeObj = datetime.now()
        log_dir = "logs/{}_{}_{}_{}_{}_{}_{}/".format(data_id, dateTimeObj.year, dateTimeObj.month, dateTimeObj.day,
                                                      dateTimeObj.hour, dateTimeObj.minute, dateTimeObj.second)

        os.makedirs(log_dir)

        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
        model.fit(x_batch,
                  y_batch,
                  batch_size=15,
                  epochs=30,
                  callbacks=[checkpoint, reduce_lr, early_stopping],
                  validation_split=0.075)

        model.save_weights(log_dir + 'trained_weights_final.h5')

    def evaluate(self, log_dir, x_batch) -> None:
        modellist = os.listdir(log_dir)
        modellist = [file for file in modellist if "val_loss" in file]
        self.__model.load_weights(log_dir + modellist[-1])
        y_batch_pred = self._model.predict(x_batch)
        y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])

        y_batch_reshape = y_batch.reshape(y_batch.shape[0], y_batch.shape[1])
        rmse_on_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))

        print("The RMSE on Training dataset {} is {}.".format(Data_id, rmse_on_train))

        x_batch_test, y_batch_test = test_batch_generator(test_FD,
                                                          sequence_length=sequence_length,
                                                          window_size=window_size)

        x_batch_test = np.expand_dims(x_batch_test, axis=4)

        model_id = 4

        model_weights = 'trained_models/{}/best_model_{}_{}.h5'.format(self.__data_id, self.__data_id, model_id)
        if not os.path.exists(model_weights):
            print("no such a weight file")
        else:
            model.load_weights(model_weights)

        y_batch_pred_test = model.predict(x_batch_test)
        rmse_on_test = np.sqrt(mean_squared_error(y_batch_pred_test, y_batch_test))

        print("The RMSE on test dataset {} is {}.".format(self.__data_id, rmse_on_test))
