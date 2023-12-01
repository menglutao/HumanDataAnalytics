import os
import pandas as pd
import subprocess

class DataLoader:
    def __init__(self):
        self.dfs = []
        self.folder_path = "dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data"

    def download_data(self):
        if os.path.exists(self.folder_path):
            print(f"Dataset folder '{self.folder_path}' already exists. Skipping download.")
        else:
            download_command = "wget -c --timeout 10 https://physionet.org/static/published-projects/accelerometry-walk-climb-drive/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0.zip -O dataset.zip"
            unzip_command = "unzip -d dataset -q dataset.zip"
            subprocess.run(download_command, shell=True, check=True)
            subprocess.run(unzip_command, shell=True, check=True)
            print("Download and extraction of dataset complete!")

    def read_files(self):
        if self.folder_path is None:
            print("Error: Folder path is not set. Please run download_data first.")
            return None
        try:
            file_names = os.listdir(self.folder_path)
            for i, file_name in enumerate(file_names):
                if file_name.endswith(".csv"):  # Check if the file is a .csv file
                    file_path = os.path.join(self.folder_path, file_name)
                    try:
                        df = pd.read_csv(file_path)
                        df.insert(0, "subject", i + 1)
                        self.dfs.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Warning: Empty file detected - {file_path}")
                    except pd.errors.ParserError as e:
                        print(f"Error reading file - {file_path}: {str(e)}")
        except FileNotFoundError:
            print("Error: Folder not found.")

    def concatenate_data(self):
        try:
            combined_df = pd.concat(self.dfs, axis=0)
            return combined_df
        except ValueError as e:
            print(f"Error concatenating data: {str(e)}")
            return None
