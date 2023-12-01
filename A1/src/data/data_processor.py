
    def calculate_vector_magnitude(self,df):
        df_transformed = df.copy()
        columns_to_calculate_VM = ["lw_x", "lw_y", "lw_z", "lh_x", "lh_y", "lh_z", "la_x", "la_y", "la_z", "ra_x", "ra_y", "ra_z"]
        position_list = ['lw','lh','la','ra']
        for i, position in zip(range(0, len(columns_to_calculate_VM), 3), position_list):
            x_col = columns_to_calculate_VM[i]
            y_col = columns_to_calculate_VM[i+1]
            z_col = columns_to_calculate_VM[i+2]
            magnitude_col = f"magnitude_{position}"
        # Calculate vector magnitude while removing sensor orientation
            df_transformed[magnitude_col] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
        # print(df_transformed.columns)
        return df_transformed

        
    def data_standardization(self,df):
        df_to_standardize = df.copy()
        scaler = StandardScaler()
        # select the columns to standardize
        # columns_to_standardize = ["lw_x", "lw_y", "lw_z", "lh_x", "lh_y", "lh_z", "la_x", "la_y", "la_z", "ra_x", "ra_y", "ra_z"]
        
        columns_to_standardize = ['magnitude_lw','magnitude_lh', 'magnitude_la', 'magnitude_ra']

        # standardize the columns
        df_to_standardize[columns_to_standardize] = scaler.fit_transform(df_to_standardize[columns_to_standardize])
        # Convert the 'time_s' column to a datetime format if it isn't already
        df_to_standardize['time_s'] = pd.to_datetime(df_to_standardize['time_s'], unit='s') 
        # Set the 'time_s' column as the index of the DataFrame
        df_to_standardize.set_index('time_s', inplace=True)
            # check if the data has been standardized
        # print(df[columns_to_standardize].mean())  # should be close to 0
        # print(df[columns_to_standardize].std())   # should be close to 1

        # Reset the index back to RangeIndex
        df_to_standardize.reset_index(inplace=True)
        # Convert 'time_s' back to its original unit
        df_to_standardize['time_s'] = df_to_standardize['time_s'].astype(int)
        print()

        return df_to_standardize

    def butter_lowpass(self, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,df, columns, cutoff, fs, order=3):
        df_filtered = df.copy()
        for column in columns:
            b, a = self.butter_lowpass(cutoff, fs, order=order)
            df_filtered[column] = filtfilt(b, a, df[column].values)
        return df_filtered

    def apply_noise_filter(self,df):
        # Assuming you want to apply a median filter and low-pass Butterworth filter
        # columns_to_apply_filter = ["lw_x", "lw_y", "lw_z", "lh_x", "lh_y", "lh_z", "la_x", "la_y", "la_z", "ra_x", "ra_y", "ra_z"]
        columns_to_apply_filter = ['magnitude_lw','magnitude_lh', 'magnitude_la', 'magnitude_ra']

        # Apply median filter to the specified columns
        df_filtered = df.copy()
        for column in columns_to_apply_filter:
            df_filtered[column] = medfilt(df[column], kernel_size=5)  # Adjust kernel size as needed

        # Apply low-pass Butterworth filter to the filtered data
        cutoff_freq = 0.2  # Corner frequency in Hz
        order = 3  # Butterworth filter order
        fs = 100  # Sample rate (assuming equally spaced samples)
        df_filtered = self.butter_lowpass_filter(df_filtered, columns_to_apply_filter, cutoff_freq, fs, order)

        return df_filtered


    def data_segmentation(self,df):
        df_new = df.copy()
        # Assuming your preprocessed data is stored in the DataFrame 'df_preprocessed'
        df_filtered = self.apply_noise_filter(df_new)

        # window_size = 3  # Size of each window in seconds
        window_size = [5.12,10.24]
        overlap = 0.5  # Overlap percentage (50%)
        sampling_rate = 100  # Sampling rate of your data (samples per second)
        #(i.e. 2.56s × 100Hz *0.5 =  128 sample ref: paper1

        # Calculate the number of data points in each window
        window_length = int(window_size[0] * sampling_rate)

        # Calculate the number of data points to shift the window by for the given overlap
        shift_length = int(window_length * overlap)

        # Initialize an empty list to store the segmented data
        segmented_data = []

        # Iterate over the data using a sliding window
        start_index = 0
        while start_index + window_length <= len(df_filtered):
            end_index = start_index + window_length
            segment = df_filtered.iloc[start_index:end_index]
            segmented_data.append(segment)
            start_index += shift_length

        # Concatenate the segmented data into a new DataFrame
        df_segmented = pd.concat(segmented_data)

        # Reset the index of the segmented DataFrame
        df_segmented.reset_index(drop=True, inplace=True)

        return df_segmented



    def statistical_extraction(self,data_processed):
        results = []
        # calculate mean,std,variance,minimum,maximun,
        mean = np.mean(data_processed)
        # Calculate standard deviation
        std = np.std(data_processed)

        # Calculate variance
        variance = np.var(data_processed)

        # Calculate minimum
        minimum = np.min(data_processed)

        # Calculate maximum
        maximum = np.max(data_processed)
        result = {
                'subject':data_processed['subject'],
                'mean': mean,
                'std': std,
                'variance': variance,
                'minimum': minimum,
                'maximum': maximum
            }
            
        results.append(result)
        results_df = pd.DataFrame(results)

        # Save  results to a CSV file
        results_df.to_csv('statistical_resultsid.id1c7e64ad.csv', index=False)

        return results
    
     # split raw dataset as train 0.6, validation 0.2, test 0.2   
    def data_train_test_split(self,combined_data):
        print(combined_data.columns)
        # combined_data.drop(columns=combined_data['index'])
        # print(combined_data.columns)

        # Split the data into train and test
        train, test = train_test_split(combined_data, test_size=0.2, random_state=42, shuffle=False)

        # Further split the training set into train and validation
        train, valid = train_test_split(train, test_size=0.25, random_state=42, shuffle=False)



        train.to_csv('train.csv')
        valid.to_csv('validation.csv')
        test.to_csv('test.csv')
        return train,valid,test








