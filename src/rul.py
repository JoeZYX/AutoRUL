import uuid
import warnings

warnings.filterwarnings("ignore")
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed
from tensorflow.keras.models import Sequential

from ECLSTM import ECLSTM1D, build_the_model, check_the_config_valid
from parameters import architecture_parameters
from utilities import *


class RemainingUsefulLife:
    """Class which predicts the remaining useful life.

    Attributes:
        data: Data to predict the remaining useful life for.
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
    def __init__(self,
                 data_id: str,
                 non_features: List[str] = None,
                 max_life: int = 120,
                 sequence_length: int = 5,
                 window_size: int = 16) -> None:
        """Inits RemainingUsefulLife."""

        # Folder name
        self.__data_id = data_id
        self.__rtf_id = None
        # self.__cycle = self.__data['cycle']
        self.__max_life = max_life
        self.__non_features = [] or non_features
        self.__model = None
        self.__model_id = None
        self.__sequence_length = sequence_length
        self.__window_size = window_size
        self.__train_FD = pd.DataFrame()
        self.__test_FD = pd.DataFrame()

        # Only for CMAPSSData
        self.__RUL_FD = None

        self.__X_batch = None
        self.__y_batch = None
        self.__number_of_sensors = None
        self.__log_dir = None
        self.__prediction = pd.DataFrame()

    # Todo: Move CMAPSSData
    # AL/FeedbackBoost/Data/CMAPs oder waterpump .../filename
    def auto_rul(self, train_fold_id: List[str], test_fold_id: List[str], rul_fold_id: Optional[List[str]]) -> None:
        if self.__data_id.lower() == 'cmapssdata':
            column_name = [
                'rtf_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
            ]
            # train_FD = []
            # test_FD = []
            # RUL_FD = []

            for fold_id in train_fold_id:
                temp = pd.read_table(f"./CMAPSSData/train_{fold_id}.txt", header=None, delim_whitespace=True)
                self.__train_FD = pd.concat([temp], ignore_index=True)

            for fold_id in test_fold_id:
                temp = pd.read_table(f"./CMAPSSData/test_{fold_id}.txt", header=None, delim_whitespace=True)
                self.__test_FD = pd.concat([temp], ignore_index=True)

            for fold_id in rul_fold_id:
                temp = pd.read_table(f"./CMAPSSData/RUL_{fold_id}.txt", header=None, delim_whitespace=True)
                self.__RUL_FD = pd.concat([temp], ignore_index=True)

            self.__train_FD.columns = column_name
            self.__test_FD.columns = column_name
            self.__rtf_id = 'rtf_id'
            self.__compute_rul()

        else:
            for fold_id in train_fold_id:
                temp = pd.read_csv(f'./TurbofanData/train_{fold_id}.csv', sep=',', decimal='.', encoding='ISO-8859-1')
                self.__train_FD = pd.concat([temp], ignore_index=True)

            for fold_id in test_fold_id:
                temp = pd.read_csv(f'./TurbofanData/test_{fold_id}.csv', sep=',', decimal='.', encoding='ISO-8859-1')
                self.__test_FD = pd.concat([temp], ignore_index=True)

                self.__train_FD.drop(self.__non_features, axis=1, inplace=True)
                self.__test_FD.drop(self.__non_features, axis=1, inplace=True)

        col_to_drop = identify_and_remove_unique_columns(self.__train_FD)
        self.__train_FD.drop(col_to_drop, axis=1, inplace=True)
        self.__test_FD.drop(col_to_drop, axis=1, inplace=True)

        mean = self.__train_FD.iloc[:, 2:-1].mean()
        std = self.__train_FD.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)

        # Training dataset
        self.__train_FD.iloc[:, 2:-1] = (self.__train_FD.iloc[:, 2:-1] - mean) / std

        # Testing dataset
        self.__test_FD.iloc[:, 2:-1] = (self.__test_FD.iloc[:, 2:-1] - mean) / std

        training_data = self.__train_FD.values
        testing_data = self.__test_FD.values

        x_train = training_data[:, 2:-1]
        y_train = training_data[:, -1]
        print("training", x_train.shape, y_train.shape)

        x_test = testing_data[:, 2:-1]
        y_test = testing_data[:, -1]
        print("testing", x_test.shape, y_test.shape)

        # sequence_length=5
        # window_size = 16
        # Prepare the training set according to the  window size and sequence_length
        self.__X_batch, self.__y_batch = batch_generator(self.__train_FD,
                                                         sequence_length=self.__sequence_length,
                                                         window_size=self.__window_size)

        self.__X_batch = np.expand_dims(self.__X_batch, axis=4)
        self.__y_batch = np.expand_dims(self.__y_batch, axis=1)
        self.__number_of_sensor = self.__X_batch.shape[-2]

        valid = check_the_config_valid(architecture_parameters, self.__window_size, self.__number_of_sensor)
        if valid:
            self.__model = build_the_model(architecture_parameters, self.__sequence_length, self.__window_size,
                                           self.__number_of_sensor)
        else:
            print("invalid configuration")
        input_data = tf.keras.layers.Input(shape=(self.__sequence_length, self.__window_size, self.__number_of_sensor,
                                                  1))
        out = self.__model(input_data)
        print(self.__model.summary())

        dateTimeObj = datetime.now()
        self.__log_dir = f"logs/{self.__data_id}_{dateTimeObj.year}_{dateTimeObj.month}_{dateTimeObj.day}_{dateTimeObj.hour}\
            _{dateTimeObj.minute}_{dateTimeObj.second}/"

        os.makedirs(self.__log_dir)

        checkpoint = ModelCheckpoint(self.__log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        self.__model.fit(
            self.__X_batch,
            self.__y_batch,
            batch_size=15,
            epochs=30,  # 30 original value
            callbacks=[  # logging,
                checkpoint, reduce_lr, early_stopping
            ],
            validation_split=0.075)
        self.__model.save_weights(self.__log_dir + 'trained_weights_final.h5')

        modellist = os.listdir(self.__log_dir)
        modellist = [file for file in modellist if "val_loss" in file]
        self.__model.load_weights(self.__log_dir + modellist[-1])

        y_batch_pred = self.__model.predict(self.__X_batch)
        y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])
        y_batch_reshape = self.__y_batch.reshape(self.__y_batch.shape[0], self.__y_batch.shape[1])
        rmse_on_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))

        print(f"The RMSE on Training dataset {self.__data_id} is {rmse_on_train}.")

    def test(self) -> None:

        x_batch_test, y_batch_test = test_batch_generator(self.__test_FD,
                                                          sequence_length=self.__sequence_length,
                                                          window_size=self.__window_size)

        x_batch_test = np.expand_dims(x_batch_test, axis=4)

        # model_id = 4
        # self.__model_id = 4
        # model_weights = f'trained_models/{self.__data_id}/best_model_{self.__data_id}_{self.__model_id}.h5'
        # if not os.path.exists(model_weights):
        #     print("no such a weight file")
        # else:
        #     model.load_weights(model_weights)

        y_batch_pred_test = self.__model.predict(x_batch_test)
        rmse_on_test = np.sqrt(mean_squared_error(y_batch_pred_test, y_batch_test))
        print(f"The RMSE on test dataset {self.__data_id} is {rmse_on_test}.")

        self.__prediction = pd.DataFrame(y_batch_pred_test, columns=['Predicted RUL'])

        indices = []  #
        rtf_id = self.__test_FD['rtf_id'].tolist()
        for idx in range(1, len(rtf_id)):
            if rtf_id[idx] > rtf_id[idx - 1]:
                indices.append(idx - 1)

            elif idx == len(rtf_id) - 1:
                indices.append(idx)

        cycle = [self.__test_FD.iloc[idx]['cycle'] for idx in indices]

        # data = {'rtf_id': list(set(self.__test_FD['rtf_id'])), 'cycle': cycle, 'Predicted RUL': list(y_batch_pred_test)}
        # results = pd.DataFrame(data=data)

        self.__prediction['rtf_id'] = list(set(self.__test_FD['rtf_id']))
        self.__prediction['cycle'] = cycle

    def __compute_rul(self) -> None:
        id = self.__rtf_id
        rul = []
        for _id in set(self.__train_FD[id]):
            trainFD_of_one_id = self.__train_FD[self.__train_FD[id] == _id]
            cycle_list = trainFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list)

            knee_point = max_cycle - self.__max_life
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                #
                if i < knee_point:
                    kink_RUL.append(self.__max_life)
                else:
                    tmp = max_cycle - i - 1
                    kink_RUL.append(tmp)
            rul.extend(kink_RUL)

        self.__train_FD["RUL_pw"] = rul

        id = self.__rtf_id
        rul = []
        for _id_test in set(self.__test_FD[id]):
            true_rul = int(self.__RUL_FD.iloc[_id_test - 1])
            testFD_of_one_id = self.__test_FD[self.__test_FD[id] == _id_test]
            cycle_list = testFD_of_one_id['cycle'].tolist()
            max_cycle = max(cycle_list) + true_rul
            knee_point = max_cycle - self.__max_life
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_RUL.append(self.__max_life)
                else:
                    tmp = max_cycle - i - 1
                    kink_RUL.append(tmp)

            rul.extend(kink_RUL)

        self.__test_FD["RUL_pw"] = rul

    def export_to_csv(self, test_fold_id: str) -> None:
        """Exports results to csv containing: rtf_id, timestamp/cycle, prediction."""
        if self.__data_id.lower() == 'cmapssdata':
            self.__prediction.to_csv(f'./CMAPSSData/RUL_pred_{test_fold_id}.csv', index=False)
        else:
            self.__prediction.to_csv(f'./TurbofanData/RUL_pred_{test_fold_id}.csv', index=False)

    def get_results(self):
        return self.__prediction.copy(deep=True)
