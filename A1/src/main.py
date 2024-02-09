# Standard Library Imports
import os
import datetime
import random as rn
import warnings
from collections import Counter
from pathlib import Path

# Data Manipulation
import pandas as pd
import numpy as np

# Machine Learning and Data Processing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
# TensorFlow and Keras
from keras.callbacks import EarlyStopping

# Local Application Imports
from data.data_loader import DataLoader
from data.data_segmentation import DataSegmentation
from utils.activity_type import ActivityType
from utils.utils import load_person_df_map, preprocess_data, preprocess_data_2, train_model

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import logging


warnings.filterwarnings('ignore')
tf.compat.v1.enable_eager_execution()

def reset_seeds():
   os.environ["PYTHONHASHSEED"] = "42"
   np.random.seed(42) 
   rn.seed(12345)
   tf.random.set_seed(1234)



def main():
    num_runs = 1
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"logs/training_log_{current_time}.txt"
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()

    # Choose one of the two methods of preprocessing
    method = "GAC"   # Grouped Activity Concatenation GAC
    if method == "DC": # Direct Concatenation
        train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped = preprocess_method_1()
    else:
        X_train,y_train,X_test,y_test = preprocess_method_2()

    models = ["simple_dense_network","simple_CNN","LSTM" ,"deep_conv_lstm_model"]


    logger.info(
    "Training Log\n"
    f"Date and Time: {datetime.datetime.now()}\n"
    f"Running the training process {num_runs} times\n\n"
    f"Seed for this run is: {42}, {12345}, {1234}\n"
    f"Preprocessing method: {method}\n"
)
    for model_name in models:
        logger.info(
            f"Training model: {model_name}\n"
    )
        for i in range(num_runs):
            tf.compat.v1.enable_eager_execution()
            reset_seeds() 
            if method == "DC":
                loss,accuracy,precision,recall,f1= train_model(model_name,train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped)#first method
            else:
                loss,accuracy,precision,recall,f1= train_model(model_name,X_train,y_train,X_test,y_test) #second method
            print("loss,accuracy,precision,recall,f1",loss,accuracy,precision,recall,f1)
            logger.info(
                f"Run {i+1}:\n"
                f"Accuracy = {accuracy}\n"
                f"Precision = {precision}\n"
                f"Recall = {recall}\n"
                f"F1 = {f1}\n"
                f"Loss = {loss}"
            )
            tf.keras.backend.clear_session()


def preprocess_method_1():
    # First method of preprocessing
    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
    
   
    data_seg = DataSegmentation(window_duration=1.28, overlap=0.5, sampling_rate=100)


    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    label_mapping = ActivityType.create_label_mapping()
    
    train_data_y_1d = np.squeeze(train_data_y)
    test_data_y_1d = np.squeeze(test_data_y)

    train_data_y_1d_mapped = np.vectorize(label_mapping.get)(train_data_y_1d)
    test_data_y_1d_mapped = np.vectorize(label_mapping.get)(test_data_y_1d)

    return train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped

def preprocess_method_2():
    WALKING = 1
    DESCENDING = 2
    ASCENDING = 3
    DRIVING = 4

    activities_list_to_consider = [WALKING, DESCENDING, ASCENDING, DRIVING]
    person_df_map = load_person_df_map(activities_list_to_consider)
    X, y= preprocess_data(person_df_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Reshape input for Conv1D
    num_features = X_train.shape[2]
    window_size=128
    X_train = X_train.reshape((-1, window_size, num_features))
    X_test = X_test.reshape((-1, window_size, num_features))
    return X_train, y_train,X_test,y_test

def preprocess_method_3():

    WALKING = 1
    DESCENDING = 2
    ASCENDING = 3
    DRIVING = 4
    activities_list_to_consider = [WALKING, DESCENDING, ASCENDING, DRIVING]
    person_df_map = load_person_df_map(activities_list_to_consider)
    X_train, y_train,X_test,y_test = preprocess_data_2(person_df_map)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test,y_test = shuffle(X_test, y_test, random_state = 42)
    return X_train, y_train,X_test,y_test

if __name__ == "__main__":
    main()
