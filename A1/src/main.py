from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
import numpy as np
import random as rn
import datetime
import os
from utils.utils import select_model,train_model,plot

from keras.callbacks import EarlyStopping
import logging
# System and File Operations
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.utils import load_person_df_map, preprocess_data

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns



# To suppress potential warnings
import warnings
warnings.filterwarnings('ignore')


tf.compat.v1.enable_eager_execution()

# tf.compat.v1.disable_eager_execution()

def reset_seeds():
   os.environ["PYTHONHASHSEED"] = "42"
   np.random.seed(42) 
   rn.seed(12345)
   tf.random.set_seed(1234)


# Constants


WALKING = 1
DESCENDING = 2
ASCENDING = 3

activities_list_to_consider = [WALKING, DESCENDING, ASCENDING]


def main():
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs"
    ]
    # 2nd way of segmenting data -> better performance
    
    person_df_map = load_person_df_map(activities_list_to_consider)
    X, y = preprocess_data(person_df_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(np.shape(X_train))
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Reshape input for Conv1D
    num_features = X_train.shape[2]
    window_size=128
    X_train = X_train.reshape((-1, window_size, num_features))
    X_test = X_test.reshape((-1, window_size, num_features))
    

    '''

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
    
   
    data_seg = DataSegmentation(window_duration=1.28, overlap=0.5, sampling_rate=100)


    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    label_mapping = ActivityType.create_label_mapping()
    

    # one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping)
    # one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)

    
    # final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)
    # final_test_y = one_hot_encoded_test_y.reshape(one_hot_encoded_test_y.shape[0],-1)



    # print("train_data_X.shape:",train_data_X.shape) #train_data_X.shape: (25823, 128, 12)
    # print("train_data_y.shape:",train_data_y.shape) #train_data_y.shape: (25823, 1)
    # print("label_mapping:",label_mapping) #label_mapping: {1: 0, 2: 1, 3: 2}
    # print("one_hot_encoded_train_y:",one_hot_encoded_train_y.shape) #one_hot_encoded_train_y: (25823, 1, 3)


    
    # # Test LSTM_CNN Model    
    # print("train_data_X.shape:",train_data_X.shape)
    # print("final_train_y.shape:",final_train_y.shape)
    # # print("y:",final_train_y)

    
    train_data_y_1d = np.squeeze(train_data_y)
    test_data_y_1d = np.squeeze(test_data_y)

    train_data_y_1d_mapped = np.vectorize(label_mapping.get)(train_data_y_1d)
    test_data_y_1d_mapped = np.vectorize(label_mapping.get)(test_data_y_1d)

    '''

    # print("1D y:",train_data_y_1d.shape)

    num_runs = 1
    log_file = "training_log29.txt"
    
    # model_name = "LSTM_CNN"
    # model_name = "Dual_LSTM"
    # model_name = "DeepConvLSTM"
    # model_name = "DeepConvLSTM2"
    model_name = "DeepConvLSTM3"

    # model_name = "simple_CNN"
    # model_name = "DeepConvLSTM4"

    with open(log_file, "w") as file:
        file.write("Training Log\n")
        file.write(f"Date and Time: {datetime.datetime.now()}\n")
        file.write(f"Running the training process {num_runs} times\n\n")
        file.write(f"Seed for this run is : {42}, {12345},{1234} \n\n")
        file.write(f"training epoch: {10} learning rate: {0.001}, batch_size = {32}, model is: {model_name} with new way of segementing data\n")
        for i in range(num_runs):
            tf.compat.v1.enable_eager_execution()
            reset_seeds() 
            # tf.compat.v1.disable_eager_execution()  # Or enable, depending on your requirement
            loss,accuracy,precision,recall,f1= train_model(model_name,X_train,y_train,X_test,y_test)
            # loss,accuracy,precision,recall,f1= train_model(model_name,train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped)
            file.write(f"Run {i+1}: Accuracy = {accuracy}\n")
            file.write(f"Run {i+1}: Precision = {precision}\n")
            file.write(f"Run {i+1}: Recall = {recall}\n")
            file.write(f"Run {i+1}: F1 = {f1}\n")
            file.write(f"Run {i+1}: Loss = {loss}\n\n")
            tf.keras.backend.clear_session()
    print(f"Training complete. Log written to {log_file}")
    


   
  


  
if __name__ == "__main__":
    main()
