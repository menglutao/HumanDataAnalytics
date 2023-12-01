import pandas as pd

class DataSegmentation:
    """
    Class for segmenting data based on window size and overlap.
    """

    def __init__(self, window_size, overlap, sampling_rate):
        """
        Initializes the DataSegmentation.

        Args:
            window_size (list): Window size for segmentation, in seconds.
            overlap (float): Overlap between windows, as a fraction.
            sampling_rate (int): Sampling rate of the data, in Hz.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate

    def __call__(self, df):
        """
        Segments the data based on initialized window size and overlap.

        Args:
            df (pd.DataFrame): DataFrame with data to segment.

        Returns:
            pd.DataFrame: Segmented DataFrame.
        """
        window_length = int(self.window_size[0] * self.sampling_rate)
        shift_length = int(window_length * self.overlap)

        segmented_data = []
        start_index = 0
        while start_index + window_length <= len(df):
            end_index = start_index + window_length
            segment = df.iloc[start_index:end_index]
            segmented_data.append(segment)
            start_index += shift_length

        df_segmented = pd.concat(segmented_data)
        df_segmented.reset_index(drop=True, inplace=True)
        return df_segmented

