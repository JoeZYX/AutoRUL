import pandas as pd
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test, target_real_feature_binary_test
from sklearn import preprocessing
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import math
import numpy as np

def identify_and_remove_unique_columns(Dataframe, rtf_id = "rtf_id", cycle_column_name = "cycle"):
    Dataframe = Dataframe.copy()
    del Dataframe[rtf_id]
    del Dataframe[cycle_column_name]
    
 
    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_to_drop = list(record_single_unique['feature'])
    Dataframe = Dataframe.drop(columns = unique_to_drop)
    

    unique_counts = Dataframe.nunique()
    record_single_unique = pd.DataFrame(unique_counts).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    record_single_unique["type"] = record_single_unique["nunique"].apply(lambda x:"real" if x>2 else "binary")
    for i in range(record_single_unique.shape[0]):
        col = record_single_unique.loc[i,"feature"]
        _type = record_single_unique.loc[i,"type"]
        if _type == "real":
            p_value = target_real_feature_real_test(Dataframe[col], Dataframe["RUL"])
        else:
            le = preprocessing.LabelEncoder()
            p_value = target_real_feature_binary_test(pd.Series(le.fit_transform(Dataframe[col])), Dataframe["RUL"])
        if p_value>0.05:
            unique_to_drop.append(col)
    
    return  unique_to_drop
	
	
def test_new_generator(test_data, sequence_length=5, window_size = 9, rtf_id ="rtf_id"):

    engine_ids = list(test_data[rtf_id].unique())
    df_new = []
    feature_number = test_data.shape[1]-3 
    for _id in set(test_data[rtf_id]):
        test_of_one_id =  test_data[test_data[rtf_id] == _id]
        if test_of_one_id.shape[0]==sequence_length+window_size-1:
            window_temp =test_of_one_id
        else:
            window_temp =test_of_one_id.iloc[1-sequence_length-window_size:,:]
        
        if _id == 1:
            df_new = window_temp
        else:
            df_new = df_new.append(window_temp)
    return df_new

def test_batch_generator(test, sequence_length=10, window_size = 10, rtf_id = "rtf_id"):
    test_data = test_new_generator(test,sequence_length=sequence_length,window_size=window_size, rtf_id= rtf_id)
    engine_ids = list(test_data[rtf_id].unique())
    index_list=[]
    temp = test_data.copy()
    feature_number =test_data.shape[1]-3
    x_shape = (len(test_data[rtf_id].unique()), sequence_length, window_size, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    y_shape = (len(test_data[rtf_id].unique()))
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)
    for _id in set(test_data[rtf_id]):
        test_of_one_id =  test_data[test_data[rtf_id] == _id]
        if test_of_one_id.shape[0]<sequence_length+window_size-1:
            for i in range(sequence_length+window_size-1-test_of_one_id.shape[0]):
                test_of_one_id = pd.concat((pd.DataFrame(test_of_one_id.iloc[0,:]).T,test_of_one_id))
            
        y_batch[_id-1] = test_of_one_id.iloc[-1, -1]
        for seq in range(sequence_length):
            
            x_batch[_id-1][seq] = test_of_one_id.iloc[seq:seq+ window_size, 2:-1].values
    
    return x_batch, y_batch

class LossHistory(keras.callbacks.Callback ):

    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type, fig_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        #if loss_type == 'epoch':

            # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig('./'+fig_name+'.jpg')
        plt.show()
        
        
def caculate_score(y_true, y_pre):
    uuts = y_true.shape[0]
    error_all = []
    y_true = y_true.flatten()
    y_pre = y_pre.flatten()
    for i in range(y_true.shape[0]):
        d = y_pre[i]-y_true[i]
        if d <0:
            s = math.exp(-d/13)-1
        else:
            s = math.exp(d/10)-1
        error_all.append(s)
        
    error = sum(error_all)
    score = error
    return score
    
    
 
def batch_generator(training_data, sequence_length=15, window_size = 15, rtf_id = "rtf_id", cycle_column_name = "cycle"):
    """
    Generator function for creating random batches of training-data
    """
    engine_ids = list(training_data[rtf_id].unique())
    temp = training_data.copy()
    for id_ in engine_ids:
        indexes = temp[temp[rtf_id] == id_].index
        traj_data = temp.loc[indexes]
        cutoff_cycle = max(traj_data[cycle_column_name]) - sequence_length - window_size + 1
        
        if cutoff_cycle<0:
            drop_range = indexes
            print("sequence_length + window_size is too large")
        else:
            cutoff_cycle_index = traj_data[cycle_column_name][traj_data[cycle_column_name] == cutoff_cycle+2].index
            drop_range = list(range(cutoff_cycle_index[0], indexes[-1] + 1))
            
        temp.drop(drop_range, inplace=True)
    indexes = list(temp.index)
    del temp
    
    feature_number = training_data.shape[1]-3

    x_shape = (len(indexes), sequence_length, window_size, feature_number)
    x_batch = np.zeros(shape=x_shape, dtype=np.float32)
    y_shape = (len(indexes))
    y_batch = np.zeros(shape=y_shape, dtype=np.float32)

    alt_index = indexes[0]
    for batch_index, index in enumerate(indexes):
        y_batch[batch_index] = training_data.iloc[index+window_size-2+sequence_length,-1]
        

        
        if index-alt_index==1 and batch_index!=0:
            temp_window = training_data.iloc[index+sequence_length-1:index+sequence_length-1 + window_size, 2:-1].values.reshape(1,window_size,-1)
            x_batch[batch_index] = np.concatenate((x_batch[batch_index-1][1:],temp_window))
        else:
            for seq in range(sequence_length):
                x_batch[batch_index][seq] = training_data.iloc[index+seq:index+seq + window_size, 2:-1].values
        alt_index = index

    
    return x_batch, y_batch   
    
def extract_RUL_per_rtf_id(df, rul_column_name = "RUL", rtf_id_column_name = "rtf_id"): 
    RUL_per_rtf_ID = list()
    rtf_ids = list()
    for rtf_id in df[rtf_id_column_name].unique():
        RUL_per_rtf_ID.append(df.loc[df[rtf_id_column_name] == rtf_id][rul_column_name].iloc[-1])
        rtf_ids.append(rtf_id)
    rul_and_rtf_id = {"rtf_id": rtf_ids, "rul_per_rtf_id": RUL_per_rtf_ID}

    return   pd.DataFrame(rul_and_rtf_id, columns=['rtf_id','rul_per_rtf_id'])