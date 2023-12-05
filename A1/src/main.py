from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader

from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
import numpy as np

def main():

    # Output classes to learn how to classify
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs",
        "Driving",
        "Clapping",
        "Non-Study Activity"
    ]

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()

    data_seg = DataSegmentation(window_duration=2.56, overlap=0.5, sampling_rate=50)
    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    label_mapping = ActivityType.create_label_mapping()
    print("label_mapping:",label_mapping)
    one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping).reshape(-1,6)
    one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)
    final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)

    print("Train Data X:",train_data_X.shape)
    print("Train Data y:",train_data_y.shape)
    # print("Test Data X:",test_data_X.shape)
    # print("Test Data y:",test_data_y.shape)
    # see the distribution of labels
    # print(Counter(one_hot_encoded_test_y.reshape(-1)))
    # print(one_hot_encoded_test_y[1167:1190])

    print(train_data_X.shape, final_train_y.shape,
          np.mean(train_data_X), np.std(train_data_X))
    






    
if __name__ == "__main__":
    main()
