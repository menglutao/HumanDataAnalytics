from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
from utils.build import Build
from utils.train import Train
from utils.performance import Performance
import numpy as np
import random as rn

tf.compat.v1.disable_eager_execution()
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(1234)

def main():

    # Output classes to learn how to classify
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs"
    ]

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
   
    data_seg = DataSegmentation(window_duration=2.56, overlap=0.5, sampling_rate=50)


    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    print("train_data_X.shape:",train_data_X.shape)

    label_mapping = ActivityType.create_label_mapping()
    print("label_mapping:",label_mapping)

    one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping)
    one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)

    final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)
    final_test_y = one_hot_encoded_test_y.reshape(one_hot_encoded_test_y.shape[0],-1)

    builder = Build(train_data_X,test_data_X,final_train_y,final_test_y)
    X, Y, accuracy, cost, optimizer, config, pred_Y = builder.build()
    trainer = Train(config,optimizer,pred_Y,cost,accuracy,test_data_X,final_test_y,train_data_X,final_train_y,X,Y)
    train_losses, train_accuracies, test_losses, test_accuracies = trainer.train()
    print("test_accuracy:",test_accuracies)
    performance = Performance(config)
    performance.plot_performance(train_losses, train_accuracies, test_losses, test_accuracies)

  
if __name__ == "__main__":
    main()
