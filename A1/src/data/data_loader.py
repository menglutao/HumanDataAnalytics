import os
import pandas as pd
import subprocess
from math import floor
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, folder_path, subject_split_ratio=0.7, random_state=42):
        self.train_dfs = []
        self.test_dfs = []
        self.folder_path = folder_path
        self.subject_split_ratio = subject_split_ratio
        self.random_state = random_state

    def download_data(self):
        if os.path.exists(self.folder_path):
            print(f"Dataset folder '{self.folder_path}' already exists. Skipping download.")
        else:
            download_command = "wget -c --timeout 10 https://physionet.org/static/published-projects/accelerometry-walk-climb-drive/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0.zip -O dataset.zip"
            unzip_command = "unzip -d dataset -q dataset.zip"
            subprocess.run(download_command, shell=True, check=True)
            subprocess.run(unzip_command, shell=True, check=True)
            print("`Download and extraction of dataset complete!")

    def read_files(self):
        if self.folder_path is None:
            print("Error: Folder path is not set. Please run download_data first.")
            return
        print(self.folder_path)
        print(os.getcwd())
        subjects = os.listdir(self.folder_path)
        num_train_subjects = floor(len(subjects) * self.subject_split_ratio)
        train_subjects, test_subjects = train_test_split(subjects, train_size=num_train_subjects, random_state=self.random_state, shuffle=True)

        for subject in subjects:
            file_path = os.path.join(self.folder_path, subject)
            try:
                df = pd.read_csv(file_path)
                df.insert(0, "subject", subject)

                if subject in train_subjects:
                    self.train_dfs.append(df)
                else:
                    self.test_dfs.append(df)
            except Exception as e:
                print(f"Error processing file '{file_path}': {str(e)}")

    def get_combined_data(self):
        combined_train_data = pd.concat(self.train_dfs, ignore_index=True) if self.train_dfs else pd.DataFrame()
        combined_test_data = pd.concat(self.test_dfs, ignore_index=True) if self.test_dfs else pd.DataFrame()
        return combined_train_data, combined_test_data






  