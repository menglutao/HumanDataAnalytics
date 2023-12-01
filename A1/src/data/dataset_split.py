from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class X_Y_Split:
    """
    Class for splitting data into X and y.
    """

    def __init__(self, target_column='activity'):
        """
        Initializes the X_Y_Split.

        Args:
            target_column (str): Name of the target column.
        """
        self.target_column = target_column

    def __call__(self, data):
        """
        Splits the data into X and y.

        Args:
            data (pd.DataFrame): DataFrame with data to split.

        Returns:
            tuple: Two DataFrames representing X and y.
        """
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        return X, y

