from data.data_loader import DataLoader
from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType

def main():

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
    train_data, test_data = data_loader.get_combined_data()


    #Train Data Shape: (6336500, 15)
    #Test Data Shape: (2845600, 15)
    print(f"Train Data Shape: {(train_data.shape)}")
    print(f"Test Data Shape: {(test_data.shape)}")



    
if __name__ == "__main__":
    main()
