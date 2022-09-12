from msilib import sequence
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
    def __init__(self,
                 train_df, 
                 test_df, 
                 test_rul, 
                 max_life: int = 2*8*6, 
                 sequence_length: int = 5, 
                 window_size: int = 16, rtf_id = 'rtf_id', cycle_column_name = 'cycle', data_id = 'kaggel_plant') -> None:
        self.__train_df = train_df
        self.__test_df = test_df
        self.__test_rul = test_rul
        self.__max_life = max_life
        self.__sequence_length = sequence_length
        self.__window_size = window_size
        self.__rtf_id = rtf_id
        self.__cycle_column_name = cycle_column_name
        self.__data_id = data_id

    def compute_piecewise_linear_rul(self):
        id= self.__rtf_id
        rul = [] 
        for _id in set(self.__train_df[id]):
            trainFD_of_one_id =  self.__train_df[self.__train_df[id] == _id]
            cycle_list = trainFD_of_one_id[self.__cycle_column_name].tolist()
            max_cycle = max(cycle_list)

            knee_point = max_cycle - self.__max_life
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                # 
                if i < knee_point:
                    kink_RUL.append(self.__max_life)
                else:
                    tmp = max_cycle-i-1 # why substracting -1 as well?
                    kink_RUL.append(tmp)
            rul.extend(kink_RUL)
        self.__train_df["RUL"] = rul 

        rul = []
        # replacement_mapping_dict needs to be automated.
        if self.__data_id == 'kaggel_plant':
            replacement_mapping_dict = {             # replacing the ids with chronoligcal 1,2,3... since the code in autorul assumes this 
                15: 1,
                16: 2
            }
            self.__test_df[id] = self.__test_df[id].replace(replacement_mapping_dict)

        for _id_test in set(self.__test_df[id]):      
            true_rul = int(self.__test_rul.iloc[_id_test - 1])
            testFD_of_one_id =  self.__test_df[self.__test_df[id] == _id_test]
            cycle_list = testFD_of_one_id[self.__cycle_column_name].tolist()
            max_cycle = max(cycle_list) + true_rul
            knee_point = max_cycle - self.__max_life
            kink_RUL = []
            for i in range(0, len(cycle_list)):
                if i < knee_point:
                    kink_RUL.append(self.__max_life)
                else:
                    tmp = max_cycle-i-1
                    kink_RUL.append(tmp)    

            rul.extend(kink_RUL)

        self.__test_df["RUL"] = rul
        

           # feature extension
    def feature_extension(self): 
        col_to_drop = identify_and_remove_unique_columns(self.__train_df, rtf_id = self.__rtf_id, cycle_column_name = self.__cycle_column_name)
        self.train_data_with_piecewise_rul = self.__train_df.drop(col_to_drop,axis = 1)
        self.test_data_with_piecewise_rul = self.__test_df.drop(col_to_drop,axis = 1)
    
    def standort_normalization(self):
        mean = self.train_data_with_piecewise_rul.iloc[:, 2:-1].mean()
        std = self.train_data_with_piecewise_rul.iloc[:, 2:-1].std()
        std.replace(0, 1, inplace=True)
        # training dataset
        self.train_data_with_piecewise_rul.iloc[:, 2:-1] = (self.train_data_with_piecewise_rul.iloc[:, 2:-1] - mean) / std

            #Testing dataset
        self.test_data_with_piecewise_rul.iloc[:, 2:-1] = (self.test_data_with_piecewise_rul.iloc[:, 2:-1] - mean) / std


    def plot_rul(self):
        self.compute_piecewise_linear_rul()
        self.feature_extension()
        self.standort_normalization()

        training_data = self.train_data_with_piecewise_rul.values
        testing_data = self.test_data_with_piecewise_rul.values

        x_train = training_data[:, 2:-1] # train data without "rul, rtf_id, cycle" columns
        y_train = training_data[:, -1] # RUL per cycle
        print("training", x_train.shape, y_train.shape)

        x_test = testing_data[:, 2:-1]
        y_test = testing_data[:, -1]
        print("testing", x_test.shape, y_test.shape)

        plt.figure(figsize=(15,2))
        plt.plot(y_train, label="train") # y_train[:500] contains cycle ruls for different engines
        plt.legend()
        plt.figure(figsize=(15,2))

        plt.plot(y_test, label="test")
        plt.legend()
        plt.figure(figsize=(15,2))
        plt.plot(x_train) # one could also restrict the plot to x_train[190] for example
        plt.title("train: " + self.__data_id )




        