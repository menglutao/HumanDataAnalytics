from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class TrainTestSplit:
    """
    Class for splitting data into train, test, and validation sets.
    """

    def __init__(self, subject_split_ratio=0.7, validation_size=0.25, random_state=42):
        """
        Initializes the TrainTestSplit.

        Args:
            subject_split_ratio (float): Proportion of subjects to include in the training split.
            validation_size (float): Proportion of training data to include in the validation split.
            random_state (int): Random state for reproducibility.
        """
        self.subject_split_ratio = subject_split_ratio
        self.validation_size = validation_size
        self.random_state = random_state

    def __call__(self, combined_data):
        """
        Splits the data into train, validation, and test sets based on subjects.

        Args:
            combined_data (pd.DataFrame): DataFrame with data to split.

        Returns:
            tuple: Three DataFrames representing train, validation, and test sets.
        """
        # Split subjects into training and testing groups
        subjects = combined_data['subject'].unique()
        train_subjects, test_subjects = train_test_split(subjects, test_size=(1 - self.subject_split_ratio), random_state=self.random_state, shuffle=True)

        # Split data based on subjects
        train_data = combined_data[combined_data['subject'].isin(train_subjects)]
        test_data = combined_data[combined_data['subject'].isin(test_subjects)]

        # Further split training data into training and validation
        train, valid = train_test_split(train_data, test_size=self.validation_size, random_state=self.random_state, shuffle=True)

        return train, valid, test_data
