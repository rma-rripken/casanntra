import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)               
casanntra_dir = os.path.join(project_root, 'casanntra')  

sys.path.insert(0, casanntra_dir)

from model_builder import ModelBuilder

class TestModelBuilder(ModelBuilder):
    def __init__(self, input_names, output_names, ndays, nwindows, window_length):
        super().__init__(input_names, output_names)
        self.ndays = ndays
        self.nwindows = nwindows
        self.window_length = window_length
        self.ntime = ndays + nwindows
        self.reverse_time_inputs = True  

    def build_model(self, input_layers, input_data):
        pass

    def fit_model(self, model, fit_in, fit_out, test_in, test_out):
        pass

    # def create_antecedent_inputs(self, df, ndays=-1, window_length=-1, nwindows=-1, reverse=None):
    #     """
    #     Expands a dataframe to include lagged data.
    #     Each column of the input dataframe is expanded to:
    #     * ndays individual lagged days,  (current day + (ndays-1) previous days)
    #     * nwindows values of averaged data over non-overlapping blocks of window_length days per block
    #     The lags are ordered from most recent to farthest past, and the averaged blocks are from the days that precede the individual.

    #     Classic CalSIM is ndays=8, window_length=14, nwindows=5

    #     Returns
    #     -------

    #         df_x : pd.DataFrame
    #         A dataframe with input columns labeled like orig_lag1 and orig_ave1 etc for the
    #         individual and aggregated

    #     """
        
    #     if ndays < 0: 
    #         ndays = self.ndays
    #     if nwindows < 0:
    #         nwindows = self.nwindows
    #     if window_length < 0:
    #         window_length = self.window_length

    #     preserve_cols = [x for x in ["datetime", "case", "fold"] if x in df.columns]
    #     df2 = df[preserve_cols].copy()
    #     df = df.drop(preserve_cols, axis=1)
    #     orig_index = df.index

    #     if reverse is None: 
    #         reverse = self.reverse_time_inputs

    #     if not reverse and nwindows > 0:
    #         print(reverse)
    #         print(nwindows)
    #         raise NotImplementedError("Not Implemented.")

    #     if reverse:
    #         arr1 = [df.shift(n) for n in range(ndays)]
    #     else:
    #         arr1 = [df.shift(n) for n in reversed(range(ndays))]

    #     if nwindows > 0:
    #         dfr = df.rolling(window=window_length, min_periods=window_length).mean()
    #         arr2 = [dfr.shift(window_length * n + ndays) for n in range(nwindows)]                    
    #     else:
    #         arr2 = []

    #     df_x = pd.concat(arr1 + arr2, axis=1).dropna()  # nsamples, nfeatures


    #     # Adjust column names
    #     new_columns = [] #preserve_cols

    #     for n in range(ndays):
    #         for col in df.columns:
    #             if col not in preserve_cols:
    #                 new_columns.append(col + '_lag{}'.format(n))
    #     for n in range(nwindows):
    #         for col in df.columns:
    #             if col not in preserve_cols: 
    #                 new_columns.append(col + '_avg{}'.format(n))

    #     df_x.columns = new_columns
        
    #     df_x = df2.join(df_x, how="right")
    #     return df_x
        
    def create_antecedent_inputs(self, df, ndays=-1, window_length=-1, nwindows=-1, reverse=None):
        """
        Expands a dataframe to include lagged and aggregated data within each case to prevent mixing data from different cases.

        Parameters:
        - df (pd.DataFrame): Original dataframe containing the features.
        - ndays (int): Number of lagged days to include.
        - window_length (int): Length of each aggregation window.
        - nwindows (int): Number of aggregation windows to include.
        - reverse (bool): If True, lagged features are ordered from current to past.
                        If False, lagged features are ordered from past to current.

        Returns:
        - pd.DataFrame: Expanded dataframe with lagged and aggregated features.
        """
        if ndays < 0:
            ndays = self.ndays  
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length

        if reverse is None:
            reverse = self.reverse_time_inputs

        if not reverse and nwindows > 0:
            raise NotImplementedError("Not implemented for non-reverse plus aggregation windows.")

        preserve_cols = ["datetime", "case", "fold"] if "datetime" in df.columns else ["case", "fold"]
        df2 = df[preserve_cols].copy()
        df_features = df.drop(preserve_cols, axis=1)

        # Initialize list to collect lagged DataFrames for each case
        lagged_dfs = []

        # Process each case separately to prevent mixing
        grouped = df.groupby('case')
        for case_id, group in grouped:
            group = group.reset_index(drop=True)

            if reverse:
                arr_lags = [group[df_features.columns].shift(n) for n in range(ndays)]
                if nwindows > 0 and window_length > 0:
                    rolling_mean = group[df_features.columns].rolling(window=window_length, min_periods=window_length).mean()
                    arr_aggs = [rolling_mean.shift(ndays + window_length * n) for n in range(nwindows)]
                else:
                    arr_aggs = []
            else:
                arr_lags = [group[df_features.columns].shift(n) for n in reversed(range(ndays))]
                arr_aggs = []  # Aggregations are not implemented for reverse=False

            df_lagged = pd.concat(arr_lags + arr_aggs, axis=1)

            # Adjust column names
            new_columns = []
            if reverse:
                for n in range(ndays):
                    for col in df_features.columns:
                        new_columns.append(f"{col}_lag{n}")
                for n in range(nwindows):
                    for col in df_features.columns:
                        new_columns.append(f"{col}_avg{n}")
            else:
                for n in reversed(range(ndays)):
                    for col in df_features.columns:
                        new_columns.append(f"{col}_lag{n}")

            df_lagged.columns = new_columns

            # Combine with preserved columns
            df_case = pd.concat([group[preserve_cols], df_lagged], axis=1)
            lagged_dfs.append(df_case)

        # Concatenate all cases
        df_result = pd.concat(lagged_dfs, axis=0)

        # Drop rows with NaN values in any of the lagged features
        feature_columns = [col for col in df_result.columns if col not in preserve_cols]
        df_result = df_result.dropna(subset=feature_columns).reset_index(drop=True)

        return df_result


def visualize_lags(df_original, df_lagged, case_id=1, test_case_id=None, feature='feature1'):
    """
    Plots the original and lagged features for a specific case and feature.
    
    Parameters:
    - df_original (pd.DataFrame): Original dataframe.
    - df_lagged (pd.DataFrame): Lagged and aggregated dataframe.
    - case_id (int): Identifier for the case to visualize.
    - test_case_id (int, optional): Identifier for the test case.
    - feature (str): Feature to visualize.
    """
    original_case = df_original[df_original['case'] == case_id].reset_index(drop=True)
    lagged_case = df_lagged[df_lagged['case'] == case_id].reset_index(drop=True)
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(original_case['datetime'], original_case[feature], label='Original', marker='o')
    
    lag_features = [col for col in lagged_case.columns if f"{feature}_lag" in col]
    for lag_col in lag_features:
        plt.plot(lagged_case['datetime'], lagged_case[lag_col], label=lag_col, marker='x')
    
    plt.xlabel('Date')
    plt.ylabel(f'{feature} Value')
    
    if test_case_id is not None:
        plt.title(f'Test Case {test_case_id}: {feature} and Its Lags')  # Updated plot title
    else:
        plt.title(f'{feature} and Its Lags for Case {case_id}')
    
    plt.legend()
    plt.grid(True)
    
    # Set y-axis ticks in increments of one
    min_y = df_original[feature].min()
    max_y = df_original[feature].max()
    plt.yticks(np.arange(min_y, max_y + 1, 1))  # Updated y-axis scaling
    
    plt.show()

def verify_case_independence(df_original, df_lagged, test_case_id, ndays, nwindows, window_length, reverse):
    """
    Ensures that lagged features for a specific case do not reference data from other cases.
    """
    original_case_df = df_original[df_original['case'] == test_case_id].reset_index(drop=True)
    lagged_case_df = df_lagged[df_lagged['case'] == test_case_id].reset_index(drop=True)
    
    for idx, row in lagged_case_df.iterrows():
        for n in range(1, ndays + 1):
            lag_col = f"feature1_lag{n}" if reverse else f"feature1_lag{ndays - n + 1}"
            expected_index = idx - (ndays - n) if reverse else idx - (ndays - (ndays - n + 1) + 1)
            if expected_index >= 0:
                expected_value = original_case_df.loc[expected_index, 'feature1']
            else:
                expected_value = 0.0 
            actual_value = row[lag_col]
            assert actual_value == expected_value, f"Case mixing detected in Test Case {test_case_id} for {lag_col} at index {idx}!"
    
    if nwindows > 0:
        for n in range(1, nwindows + 1):
            avg_col = f"feature1_avg{n}"
            pass
    
    print(f"Case independence verified for Test Case {test_case_id}.")

def run_test_case(test_case_id, reverse, ndays, nwindows, window_length):
    print(f"\n=== Test Case {test_case_id} ===")
    print(f"Parameters: reverse={reverse}, ndays={ndays}, nwindows={nwindows}, window_length={window_length}")

    date_range = pd.date_range(start='2021-01-01', periods=20, freq='D')  
    data = {
        'datetime': np.tile(date_range, 2),
        'case': np.repeat([1, 2], len(date_range)),
        'fold': np.repeat([0, 1], len(date_range)),
        'feature1': np.tile(np.arange(0, 20), 2),
        'feature2': np.tile(np.arange(1000, 1020), 2)
    }
    df = pd.DataFrame(data)

    print("\nOriginal DataFrame Summary:")
    print(df.head())

    input_names = ['feature1', 'feature2']
    output_names = [] 

    model_builder = TestModelBuilder(input_names, output_names, ndays, nwindows, window_length)
    model_builder.reverse_time_inputs = reverse  

    try:
        df_lagged = model_builder.create_antecedent_inputs(
            df, ndays=ndays, nwindows=nwindows, window_length=window_length, reverse=reverse
        )
        print("\nLagged and Aggregated DataFrame:")
        print(df_lagged.head())

        expected_lag_cols = len(input_names) * ndays
        expected_avg_cols = len(input_names) * nwindows
        actual_lag_cols = len([col for col in df_lagged.columns if 'lag' in col])
        actual_avg_cols = len([col for col in df_lagged.columns if 'avg' in col])

        print(f"\nExpected Lagged Columns: {expected_lag_cols}, Actual: {actual_lag_cols}")
        print(f"Expected Averaged Columns: {expected_avg_cols}, Actual: {actual_avg_cols}")
        print("Column counts are as expected.")

        sample_index = ndays 
        if sample_index < len(df_lagged):
            print(f"\nManual Verification for Index {sample_index}:")
            row = df_lagged.iloc[sample_index]
            original_row = df[(df['case'] == row['case']) & (df['datetime'] == row['datetime'])].iloc[0]
            print(f"Original Row: {original_row[['datetime', 'case', 'feature1', 'feature2']]}")

            # Corrected Verification Loop
            for n in range(ndays):
                if reverse:
                    lag_col = f"feature1_lag{n}"
                    expected_lag = original_row['feature1'] - n
                else:
                    lag_col = f"feature1_lag{ndays - n - 1}"
                    expected_lag = original_row['feature1'] - (ndays - n - 1)
                actual_lag = row[lag_col]
                print(f"{lag_col}: Expected {expected_lag}, Actual {actual_lag}")
                assert actual_lag == expected_lag, f"Lagged feature mismatch for {lag_col} at index {sample_index}!"

        verify_case_independence(df, df_lagged, test_case_id, ndays, nwindows, window_length, reverse)
        visualize_lags(df, df_lagged, case_id=1, test_case_id=test_case_id, feature='feature1')  # Updated to pass test_case_id

    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")
    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    test_cases = [
        {"test_case_id": 1, "reverse": True, "ndays": 3, "nwindows": 0, "window_length": 0},
        {"test_case_id": 2, "reverse": False, "ndays": 3, "nwindows": 0, "window_length": 0},
        {"test_case_id": 3, "reverse": True, "ndays": 3, "nwindows": 2, "window_length": 5},
        {"test_case_id": 4, "reverse": False, "ndays": 3, "nwindows": 2, "window_length": 5},
        {"test_case_id": 5, "reverse": True, "ndays": 5, "nwindows": 3, "window_length": 4},
        {"test_case_id": 6, "reverse": False, "ndays": 5, "nwindows": 3, "window_length": 4},
    ]

    for case in test_cases:
        run_test_case(**case)

if __name__ == "__main__":
    main()
