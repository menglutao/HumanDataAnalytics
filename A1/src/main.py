from data.data_loader import DataHandler
from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf


train_path = '../src/train.csv'
valid_path = '../src/validation.csv'
test_path = '../src/test.csv'
combined_path = '../src/combined_file.csv'

def main():
    # print(tf.__version__)
    

    # raw_folder_path = '../data/raw/physionet.org/files/accelerometry-walk-climb-drive/1.0.0/raw_accelerometry_data'
    # data_handler = DataHandler(raw_folder_path)

    # data_handler.read_files()
    # combined_df = data_handler.concatenate_data()
    # adding_vm_df = data_handler.calculate_vector_magnitude(combined_df)
    # data_after_standardize= data_handler.data_standardization(adding_vm_df)
    # data_final = data_handler.data_segmentation(data_after_standardize)

    # data_handler.statistical_extraction(data_final)

    # train,test,valid = data_handler.data_train_test_split(data_final)
    
    # # print(f'train {train.shape} , validation {valid.shape} , test {test.shape}')
    # print(f'data_final {data_final.shape} , data_final.columns: {data_final.columns}')

    train_df = read_data(train_path)
    H,W,N,input_tensor  = make_input_data(train_df)

    # Instantiate the model
    model = CNNModel(N,H,W,6)
    # # Pass the input tensor through the model
    output = model(input_tensor)
    # print('output:',output)




if __name__ == "__main__":
    main()
