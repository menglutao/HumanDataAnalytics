import numpy as np

class StatisticalExtraction:
    """
    Class for extracting statistical features from segmented data.
    """

    def __call__(self, data_processed):
        """
        Extracts statistical features from each segment in the data.

        Args:
            data_processed (pd.DataFrame): DataFrame containing segmented data.

        Returns:
            pd.DataFrame: DataFrame containing extracted features.
        """
        results = []
        for _, group in data_processed.groupby('subject'):
            mean = np.mean(group)
            std = np.std(group)
            variance = np.var(group)
            minimum = np.min(group)
            maximum = np.max(group)
            result = {
                'subject': group['subject'].iloc[0],
                'mean': mean,
                'std': std,
                'variance': variance,
                'minimum': minimum,
                'maximum': maximum
            }
            results.append(result)
        
        return pd.DataFrame(results)
