from data.data_loader import DataLoader
from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
from data.dataset_split import X_Y_Split

def main():

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
    train_data, test_data = data_loader.get_combined_data()


    #Train Data Shape: (6336500, 15)
    #Test Data Shape: (2845600, 15)
    print(f"Train Data Shape: {(train_data.shape)}")
    print(f"Test Data Shape: {(test_data.shape)}")
    # print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("Train Data:",train_data.columns)
    splitter = X_Y_Split(target_column='activity')

    # Split the combined train data
    X_train, y_train = splitter(train_data)
    # Split the combined test data
    X_test, y_test = splitter(test_data)
    print("X_train:",X_train.shape)
    print("y_train:",y_train.shape)
    print("X_test:",X_test.shape)
    print("y_test:",y_test.shape)



    
if __name__ == "__main__":
    main()
