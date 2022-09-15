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

from parameters import architecture_parameters

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
                 max_life: int = 2*8*60, 
                 sequence_length: int = 5, 
                 window_size: int = 16, rtf_id = 'rtf_id', cycle_column_name = 'cycle', data_id = 'kaggel_plant', epochs: int = 2) -> None:
        self.__train_df = train_df
        self.__test_df = test_df
        self.__test_rul = test_rul
        self.__max_life = max_life
        self.__sequence_length = sequence_length
        self.__window_size = window_size
        self.__rtf_id = rtf_id
        self.__cycle_column_name = cycle_column_name
        self.__data_id = data_id
        self.__epochs = epochs


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
        replacement_mapping_dict = {}
        for i in range(len(self.__test_df[self.__rtf_id].unique())):
            replacement_mapping_dict[self.__test_df[self.__rtf_id].unique()[i]] = i + 1       # replacing the ids with chronoligcal 1,2,3... since the code in autorul assumes this 
            
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
        self.__train_data_with_piecewise_rul = self.__train_df.drop(col_to_drop,axis = 1)
        self.__test_data_with_piecewise_rul = self.__test_df.drop(col_to_drop,axis = 1)
    
    def standard_normalization(self):
        # normalizing the training feautures
        self.__train_features = self.__train_data_with_piecewise_rul.loc[:,~self.__train_data_with_piecewise_rul.columns.isin([self.__cycle_column_name, self.__rtf_id, 'RUL'])]
        mean = self.__train_features.mean()
        std = self.__train_features.std()
        std.replace(0, 1, inplace=True)
        # training dataset
        self.__train_features = (self.__train_features - mean) / std
        # adding RUL and cycle columns back to the normalized features
        self.__train_features.loc[:, [self.__rtf_id ,self.__cycle_column_name, 'RUL']] = self.__train_data_with_piecewise_rul[[self.__rtf_id,self.__cycle_column_name, 'RUL']]
        self.__train_data_with_piecewise_rul = self.__train_features
        rtf_id_column = self.__train_data_with_piecewise_rul.pop(self.__rtf_id)
        cycle_column = self.__train_data_with_piecewise_rul.pop(self.__cycle_column_name)
        self.__train_data_with_piecewise_rul.insert(0, self.__rtf_id, rtf_id_column)
        self.__train_data_with_piecewise_rul.insert(1,self.__cycle_column_name, cycle_column)
            #Testing dataset
        self.__test_features = self.__test_data_with_piecewise_rul.loc[:,~self.__test_data_with_piecewise_rul.columns.isin([self.__cycle_column_name, self.__rtf_id, 'RUL'])]

        self.__test_features = (self.__test_features - mean) / std
        self.__test_features.loc[:, [self.__rtf_id ,self.__cycle_column_name, 'RUL']] = self.__test_data_with_piecewise_rul[[self.__rtf_id,self.__cycle_column_name, 'RUL']]


        self.__test_data_with_piecewise_rul = self.__test_features
        rtf_id_column = self.__test_data_with_piecewise_rul.pop(self.__rtf_id)
        cycle_column = self.__test_data_with_piecewise_rul.pop(self.__cycle_column_name)
        self.__test_data_with_piecewise_rul.insert(0, self.__rtf_id, rtf_id_column)
        self.__test_data_with_piecewise_rul.insert(1,self.__cycle_column_name, cycle_column)


    def plot_rul(self):

        training_data = self.__train_data_with_piecewise_rul.values
        testing_data = self.__test_data_with_piecewise_rul.values
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



    def batch_generation(self): 
        # Prepare the training set according to the  window size and sequence_length
        self.__x_batch, self.__y_batch =batch_generator(self.__train_data_with_piecewise_rul,sequence_length=self.__sequence_length,window_size = self.__window_size)
        self.__x_batch = np.expand_dims(self.__x_batch, axis=4)
        self.__y_batch = np.expand_dims(self.__y_batch, axis=1)
        self.__number_of_sensor = self.__x_batch.shape[-2]

    def train_model(self): 
        valid = check_the_config_valid(architecture_parameters ,self.__window_size,self.__number_of_sensor)
        if valid:
            self.__model = build_the_model(architecture_parameters, self.__sequence_length, self.__window_size, self.__number_of_sensor)
        else: 
            print("invalid configuration")
        input_data = tf.keras.layers.Input(shape=(self.__sequence_length, self.__window_size, self.__number_of_sensor,1))
        out = self.__model(input_data)
        print(self.__model.summary())

        dateTimeObj = datetime.now()

        self.__log_dir = "logs/{}_{}_{}_{}_{}_{}_{}/".format(self.__data_id,
                                                    dateTimeObj.year,
                                                    dateTimeObj.month,
                                                    dateTimeObj.day,
                                                    dateTimeObj.hour,
                                                    dateTimeObj.minute,
                                                    dateTimeObj.second)


        os.makedirs(self.__log_dir)

        checkpoint = ModelCheckpoint(self.__log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


        # if you have enough time budget, you can set a large epochs and large patience

         
        self.__model.fit(self.__x_batch,self.__y_batch, 
                batch_size=15, 
                epochs=self.__epochs, 
                callbacks=[#logging, 
                            checkpoint, 
                            reduce_lr, 
                            early_stopping],
                validation_split=0.075)
        self.__model.save_weights(self.__log_dir + 'trained_weights_final.h5')


    def evaluate(self):
        x_batch_test, y_batch_test =  test_batch_generator(self.__test_data_with_piecewise_rul, sequence_length=self.__sequence_length, window_size = self.__window_size)
        x_batch_test = np.expand_dims(x_batch_test, axis=4)

        modellist = os.listdir(self.__log_dir)
        modellist = [file for file in modellist if "val_loss" in file]

        self.__model.load_weights(self.__log_dir+modellist[-1])
    
        # performance on training dataset
        y_batch_pred = self.__model.predict(self.__x_batch)

        y_batch_pred = y_batch_pred.reshape(y_batch_pred.shape[0], y_batch_pred.shape[1])
        y_batch_reshape = self.__y_batch.reshape(self.__y_batch.shape[0], self.__y_batch.shape[1])
        rmse_on_train = np.sqrt(mean_squared_error(y_batch_pred, y_batch_reshape))

        print("The RMSE on Training dataset {} is {}.".format(self.__data_id,rmse_on_train))

        # performance on test dataset
        y_batch_pred_test = self.__model.predict(x_batch_test)
        rmse_on_test = np.sqrt(mean_squared_error(y_batch_pred_test, y_batch_test))
        print("The RMSE on test dataset {} is {}.".format(self.__data_id,rmse_on_test))

    def auto_rul(self): 
        RemainingUsefulLife.compute_piecewise_linear_rul(self)
        RemainingUsefulLife.feature_extension(self)
        RemainingUsefulLife.standard_normalization(self)
        RemainingUsefulLife.plot_rul(self)
        RemainingUsefulLife.batch_generation(self)
        RemainingUsefulLife.train_model(self)
        RemainingUsefulLife.evaluate(self)
        
    
        
   
