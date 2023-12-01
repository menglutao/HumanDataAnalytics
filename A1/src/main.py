from data.data_loader import DataLoader
from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf

def main():
    dataloader = DataLoader()
    dataloader.download_data()
    dataloader.read_files()
    combine_df = dataloader.concatenate_data()
    print(combine_df.head(10))
if __name__ == "__main__":
    main()
