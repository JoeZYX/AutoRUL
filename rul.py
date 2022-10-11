from enum import auto
from itertools import cycle
from msilib import sequence
from random import triangular
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
from typing import Tuple

class RemainingUsefulLife: 
        
    def __init__(self,
                 train_df, 
                 test_df, 
                 test_rul_per_rtf_id = None,
                 train_rul_per_rtf_id = None,
                 max_life: int = 120, 
                 sequence_length: int = 5, 
                 window_size: int = 16, rtf_id = 'rtf_id', cycle_column_name = 'cycle', data_id = 'kaggel_plant', epochs: int = 30, path_to_trained_model: str = None) -> None:


        self.__train_df = train_df.copy()
        self.__test_df = test_df.copy()
        self.__test_rul_per_rtf_id = test_rul_per_rtf_id
        self.__train_rul_per_rtf_id = train_rul_per_rtf_id
        self.__max_life = max_life
        self.__sequence_length = sequence_length
        self.__window_size = window_size
        self.__rtf_id = rtf_id
        self.__cycle_column_name = cycle_column_name
        self.__data_id = data_id
        self.__epochs = epochs
        self.__log_dir = path_to_trained_model

    # storing the string in a variable, since it is often used in the code
    rul_pw = "RUL_pw"

    # function to calculate the piecewise rul for both training and test data
    def compute_piecewise_linear_rul(self)->Tuple[pd.DataFrame,pd.DataFrame]:
        """_summary_

        Returns:
            Tuple[pd.DataFrame,pd.DataFrame]: _description_
        """
        id= self.__rtf_id
        datasets = [self.__train_df, self.__test_df]
        for data in datasets: # since the process for rul calculation is the same, looping through train and test data to calculate pw_rul for each of them 
            rul = [] 

            # replacing the rtf_ids
            replacement_mapping_dict = {}
            for i in range(len(data[id].unique())):
                replacement_mapping_dict[int(data[id].unique()[i])] = int(i + 1)       # replacing the unique rtf_ids with chronoligcal 1,2,3... since the code in autorul assumes this 
            data[id] = data[id].replace(replacement_mapping_dict)  # notice: we are just replacing the UNIQUE values, we are NOT restructuring the whole column from 1-n

            # pw_rul calculation
            for _id in set(data[id]): # looping through unique rtf_ids
                trainFD_of_one_id =  data[data[id] == _id]
                cycle_list = trainFD_of_one_id[self.__cycle_column_name].tolist()
         
                # a plot just for debugging purposes...   --> shows anomalies in the cycle_list
                cycle_list_ = pd.DataFrame({'cycle': cycle_list})
                plot_cycle_anomalie(cycle_list_, _id)
                
                # getting the true_rul value (at the last cycle) for an rtf_id 
                if data.equals(self.__train_df):
                    if self.__train_rul_per_rtf_id is not None:
                        true_rul = int(self.__train_rul_per_rtf_id.iloc[_id-1])
                    else: 
                        true_rul = 0
                elif data.equals(self.__test_df):
                    if self.__test_rul_per_rtf_id is not None:
                        true_rul = int(self.__test_rul_per_rtf_id.iloc[_id-1])
                    else: 
                        true_rul = 0                    

                # calculating pw_rul 
                pw_rul_for_each_rtf_id = [true_rul] 
                for i in (range(1, len(cycle_list))[::-1]): 
                    cycle_substract = cycle_list[i] - cycle_list[i-1] 
                    if i == len(cycle_list) - 1:   
                        if cycle_substract + true_rul >= self.__max_life:
                            pw_rul_for_each_rtf_id.append(self.__max_life)
                        else:
                            pw_rul_for_each_rtf_id.append(cycle_substract + true_rul)
                    else:
                        if cycle_substract + pw_rul_for_each_rtf_id[-1] >= self.__max_life:
                            pw_rul_for_each_rtf_id.append(self.__max_life)
                        else:
                            pw_rul_for_each_rtf_id.append(cycle_substract + pw_rul_for_each_rtf_id[-1])
                
                rul.extend(pw_rul_for_each_rtf_id[::-1])
            data[RemainingUsefulLife.rul_pw] = rul

        # The following is the old (based on Zhou et al. logic) implementation of the pw_rul function

        #     if self.__train_rul_per_rtf_id is not None:
        #         true_rul = int(self.__train_rul_per_rtf_id.iloc[_id - 1])
        #         max_cycle = len(cycle_list) + true_rul
        #     else:
        #         max_cycle = len(cycle_list)
        #     if self.__max_life > max_cycle: 
        #         raise ValueError("Paramater max_life is too large. Try a smaller one")
        #     knee_point = max_cycle - self.__max_life
        #     kink_RUL = []
        #     for i in range(0, len(cycle_list)):
        #         if i < knee_point:
        #             kink_RUL.append(self.__max_life)
        #         else:
        #             tmp = max_cycle-i-1 
        #             kink_RUL.append(tmp)
        #     rul.extend(kink_RUL)
        # self.__train_df[RemainingUsefulLife.rul_pw] = rul 

        # rul = []

        # for _id_test in set(self.__test_df[id]):      
        #     true_rul = int(self.__test_rul_per_rtf_id.iloc[_id_test - 1])
        #     testFD_of_one_id =  self.__test_df[self.__test_df[id] == _id_test]
        #     cycle_list = testFD_of_one_id[self.__cycle_column_name].tolist()
        #     # for i in range(len(cycle_list)):  # in case some some cycles are missing, all of the cycles will be rearranged chronoligically from 1 up to len(cycle_list) 
        #     #     cycle_list[i] = i + 1
        #     max_cycle = len(cycle_list) + true_rul
        #     if self.__max_life > max_cycle: 
        #         raise ValueError("Paramater max_life is too large. Try a smaller one")
                
        #     knee_point = max_cycle - self.__max_life
        #     kink_RUL = []
        #     for i in range(0, len(cycle_list)):
        #         if i < knee_point:
        #             kink_RUL.append(self.__max_life)
        #         else:
        #             tmp = max_cycle-i-1
        #             kink_RUL.append(tmp)    

        #     rul.extend(kink_RUL)

        # self.__test_df[RemainingUsefulLife.rul_pw] = rul
        # # train_df = self.__train_df.copy()
        return self.__train_df, self.__test_df
        

           # feature extension
    def feature_extension(self): 
        self.__train_df, self.__test_df =  RemainingUsefulLife.compute_piecewise_linear_rul(self)
        col_to_drop = identify_and_remove_unique_columns(self.__train_df, rtf_id = self.__rtf_id, cycle_column_name = self.__cycle_column_name)
        self.__train_data_with_piecewise_rul = self.__train_df.drop(col_to_drop,axis = 1)
        self.__test_data_with_piecewise_rul = self.__test_df.drop(col_to_drop,axis = 1)
        return self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul
    
    def standard_normalization(self):
        self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul = RemainingUsefulLife.feature_extension(self)
        # normalizing the training feautures
        self.__train_features = self.__train_data_with_piecewise_rul.loc[:,~self.__train_data_with_piecewise_rul.columns.isin([self.__cycle_column_name, self.__rtf_id, RemainingUsefulLife.rul_pw])]
        mean = self.__train_features.mean()
        std = self.__train_features.std()
        std.replace(0, 1, inplace=True)
        # training dataset
        self.__train_features = (self.__train_features - mean) / std
        # adding RUL and cycle columns back to the normalized features
        self.__train_features.loc[:, [self.__rtf_id ,self.__cycle_column_name, RemainingUsefulLife.rul_pw]] = self.__train_data_with_piecewise_rul[[self.__rtf_id,self.__cycle_column_name, RemainingUsefulLife.rul_pw]]
        self.__train_data_with_piecewise_rul = self.__train_features
        rtf_id_column = self.__train_data_with_piecewise_rul.pop(self.__rtf_id)
        cycle_column = self.__train_data_with_piecewise_rul.pop(self.__cycle_column_name)
        self.__train_data_with_piecewise_rul.insert(0, self.__rtf_id, rtf_id_column)
        self.__train_data_with_piecewise_rul.insert(1,self.__cycle_column_name, cycle_column)


        #Testing dataset
        self.__test_features = self.__test_data_with_piecewise_rul.loc[:,~self.__test_data_with_piecewise_rul.columns.isin([self.__cycle_column_name, self.__rtf_id, RemainingUsefulLife.rul_pw])]

        self.__test_features = (self.__test_features - mean) / std
        self.__test_features.loc[:, [self.__rtf_id ,self.__cycle_column_name,RemainingUsefulLife.rul_pw]] = self.__test_data_with_piecewise_rul[[self.__rtf_id,self.__cycle_column_name,RemainingUsefulLife.rul_pw]]

        self.__test_data_with_piecewise_rul = self.__test_features
        rtf_id_column = self.__test_data_with_piecewise_rul.pop(self.__rtf_id)
        cycle_column = self.__test_data_with_piecewise_rul.pop(self.__cycle_column_name)
        self.__test_data_with_piecewise_rul.insert(0, self.__rtf_id, rtf_id_column)
        self.__test_data_with_piecewise_rul.insert(1,self.__cycle_column_name, cycle_column)

        return self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul

    def plot_rul(self): # plotting the
        
        self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul = RemainingUsefulLife.standard_normalization(self)
        training_data = self.__train_data_with_piecewise_rul.values
        testing_data = self.__test_data_with_piecewise_rul.values
        x_train = training_data[:, 2:-1] # train data without "rul, rtf_id, cycle" columns
        y_train = training_data[:, -1] # RUL per cycle
        print("training", x_train.shape, y_train.shape)

        x_test = testing_data[:, 2:-1]
        y_test = testing_data[:, -1]
        print("testing", x_test.shape, y_test.shape)

        plt.figure(figsize=(15,2))
        plt.plot(y_train, label="train") # y_train contains cycle ruls for different engines
        plt.ylabel(RemainingUsefulLife.rul_pw)
        plt.title("Piecewise RUL for different RTF Ids")
        plt.legend()
        plt.figure(figsize=(15,2))

        plt.plot(y_test, label="test")
        plt.ylabel(RemainingUsefulLife.rul_pw)
        plt.title("Piecewise RUL for different RTF Ids")
        plt.legend()
        plt.figure(figsize=(15,2))
        plt.plot(x_train) # one could also restrict the plot to x_train[190] for example
        plt.title("train: " + self.__data_id )
        plt.show()

    def build_model(self): 
                ########## batch generation for training data ##########
        self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul = RemainingUsefulLife.standard_normalization(self)
        # Prepare the training set according to the  window size and sequence_length
        self.__x_batch, self.__y_batch =batch_generator(self.__train_data_with_piecewise_rul,sequence_length=self.__sequence_length,window_size = self.__window_size, rtf_id = self.__rtf_id, cycle_column_name = self.__cycle_column_name)
        self.__x_batch = np.expand_dims(self.__x_batch, axis=4)
        self.__y_batch = np.expand_dims(self.__y_batch, axis=1)
        self.__number_of_sensor = self.__x_batch.shape[-2]

        ############ Build model ############
        valid = check_the_config_valid(architecture_parameters,self.__window_size,self.__number_of_sensor)
        if valid:
            self.__model = build_the_model(architecture_parameters, self.__sequence_length, self.__window_size, self.__number_of_sensor)
        else: 
            print("invalid configuration")
        input_data = tf.keras.layers.Input(shape=(self.__sequence_length, self.__window_size, self.__number_of_sensor,1))
        out = self.__model(input_data)
        print(self.__model.summary())



    def train_model(self): 

        RemainingUsefulLife.build_model(self)

        dateTimeObj = datetime.now()
        # creating a folder to store the differnt models in later on
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
        # self.__train_model = True # simple checker to check if a new model is being trained 
        # else: 
        #     self.__train_model = False # if a given model is supposed to be executed
        #     print(f"can not train a model if a path to a model is given. If a new model training is desired, leave the paramter {self.__log_dir} out.") 
        

    def auto_rul(self):
        self.__train_data_with_piecewise_rul, self.__test_data_with_piecewise_rul = RemainingUsefulLife.standard_normalization(self)

        x_batch_test, y_batch_test =  test_batch_generator(self.__test_data_with_piecewise_rul, sequence_length=self.__sequence_length, window_size = self.__window_size, rtf_id= self.__rtf_id)
        x_batch_test = np.expand_dims(x_batch_test, axis=4)

        if self.__log_dir is None:
            RemainingUsefulLife.train_model(self)
            modellist = os.listdir(self.__log_dir)
            modellist = [file for file in modellist if "val_loss" in file]
            self.__model.load_weights(self.__log_dir + modellist[-1])
        # loading the weights of the given model (user input)
        else:
            model_weights = self.__log_dir
            if not os.path.exists(model_weights):
                print("no such a weight file")
            else:
                RemainingUsefulLife.build_model(self)
                self.__model.load_weights(model_weights)

        # # loading the weights of the current model which was just trained and saved
        # if self.__train_model == True: 
        #     modellist = os.listdir(self.__log_dir)
        #     modellist = [file for file in modellist if "val_loss" in file]
            
        #     self.__model.load_weights(self.__log_dir + modellist[-1])
        # # loading the weights of the given model (user input)
        # else:
        #     model_weights = self.__log_dir
        #     if not os.path.exists(model_weights):
        #         print("no such a weight file")
        #     else:
        #         self.__model.load_weights(model_weights)
    
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
        y_batch_pred_test_df = pd.DataFrame(y_batch_pred_test, columns= ["y_batch_pred_test"])
        y_batch_test = pd.DataFrame(y_batch_test, columns= ["y_batch_test"])

        y_batch_pred_df = pd.DataFrame(y_batch_pred, columns= ["y_batch_pred_train"])
        y_batch_train = pd.DataFrame(y_batch_reshape, columns= ["y_batch_train"])

        # returning the prediction results and the expected results for test and train data respectively 
        return pd.concat([y_batch_test, y_batch_pred_test_df], axis = 1, join = "inner"), pd.concat([y_batch_train, y_batch_pred_df], axis = 1, join = "inner")


    # def plot_pred(self): 
    #     RemainingUsefulLife.auto_rul(self)



    # def auto_rul(self): # function which simply excecutes all other functions
    #     RemainingUsefulLife.compute_piecewise_linear_rul(self)
    #     RemainingUsefulLife.feature_extension(self)
    #     RemainingUsefulLife.standard_normalization(self)
    #     RemainingUsefulLife.plot_rul(self)
    #     RemainingUsefulLife.train_model(self)
    #     return RemainingUsefulLife.evaluate(self)
        
    
        
   
