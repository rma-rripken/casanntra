from casanntra.antecedent_inputs import *
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import tensorflow as tf
from numpy.testing import assert_allclose

# LSTM Format Test
def test_lstm_format():
    """
    Test the create_antecedent_inputs function in the LSTM format.
    We expect 2 rows when using ndays=84 and nwindow=0 on a time series with 85 values.
    """
    # Create a time series with 85 values
    test_data = pd.DataFrame(np.arange(85), columns=["feature1"], index=pd.date_range('2023-01-01', periods=85))

    # Apply create_antecedent_inputs with ndays=84, window_size=1, nwindows=0
    # This should return 2 rows (for the full history of t=84 and t=85), each with 84 columns
    output = create_antecedent_inputs(test_data, ndays=84, window_size=1, nwindows=0)
    
    # Check that we have 2 rows and 84 columns (one per lagged day, including t)
    assert output.shape == (2, 84), f"Expected shape (2, 84), but got {output.shape}"

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import tensorflow as tf
from numpy.testing import assert_allclose
import pytest  # Optional, but recommended for pytest usage

# Updated check for congruency between input size and required days + windows
def validate_input_length(length, ndays, window_size, nwindows):
    required_length = ndays + nwindows * window_size
    if length < required_length:
        raise ValueError(f"Input length {length} is too short. It must be at least {required_length} "
                         f"to support ndays={ndays}, window_size={window_size}, nwindows={nwindows}.")

# Assuming create_antecedent_inputs is already defined (from your original function)
def create_antecedent_inputs(df, ndays=14, window_size=14, nwindows=5):
    """
    Original function that processes the input dataframe using pandas shifts and rolling means.
    This simulates the antecedent feature creation for machine learning models.
    
    Args:
    df : pd.DataFrame
        Input time-series data.
    ndays : int
        Number of individual days to retain.
    window_size : int
        Size of the aggregation window (e.g., 14 days).
    nwindows : int
        Number of windows to aggregate.
    
    Returns:
    pd.DataFrame with individual day features and rolling window averages.
    """
    # Validate that the input size is adequate
    validate_input_length(len(df), ndays, window_size, nwindows)
    
    arr1 = [df.shift(n) for n in range(ndays)]
    dfr = df.rolling(str(window_size) + 'D', min_periods=window_size).mean()
    arr2 = [dfr.shift(periods=(window_size * n + ndays)) for n in range(nwindows)]
    return pd.concat(arr1 + arr2, axis=1).dropna()



# LSTM Format Test
def test_lstm_format():
    """
    Test the create_antecedent_inputs function in the LSTM format.
    We expect 2 rows when using ndays=84 and nwindow=0 on a time series with 85 values.
    """
    # Create a time series with 85 values
    test_data = pd.DataFrame(np.arange(85), columns=["feature1"], index=pd.date_range('2023-01-01', periods=85))

    # Apply create_antecedent_inputs with ndays=84, window_size=1, nwindows=0
    # This should return 2 rows (for the full history of t=84 and t=85), each with 84 columns
    output = create_antecedent_inputs(test_data, ndays=84, window_size=1, nwindows=0)
    
    # Check that we have 2 rows and 84 columns (one per lagged day, including t)
    assert output.shape == (2, 84), f"Expected shape (2, 84), but got {output.shape}"

# MLP Format Test (with batch size of 2)
def test_mlp_format():
    """
    Test the full procedure for the MLP format using create_antecedent_layer_with_conv.
    We expect a batch size of 2 with the next dimension of ndays + nwindows = 19.
    """
    # Create a time series with 84 values (for easy checking, use range(84))
    test_data = pd.DataFrame(np.arange(85), columns=["feature1"], index=pd.date_range('2023-01-01', periods=85))

    # Apply create_antecedent_inputs with ndays=84, nwindow=0 to generate a batch of 2
    lstm_data = create_antecedent_inputs(test_data, ndays=84, window_size=1, nwindows=0)
    
    # Reshape the data for the Keras Input layer (batch size 2, 84 time steps, 1 feature)
    lstm_data_np = lstm_data.values.reshape(2, 84, 1)  # Reshape to (batch_size, time_steps, features)

    # Define the Keras input layer and apply the antecedent layer transformation
    input_layer = layers.Input(shape=(84, 1))  # The input now has 84 time steps and 1 feature
    antecedent_layer = create_antecedent_layer_with_conv(input_layer, ndays=14, window_size=14, nwindows=5)
    model = models.Model(inputs=input_layer, outputs=antecedent_layer)

    # Predict the output using the Keras model
    keras_output = model.predict(lstm_data_np)
    
    # Check that the output has the correct shape (batch_size=2, ndays + nwindows = 19)
    assert keras_output.shape == (2, 19), f"Expected shape (2, 19), but got {keras_output.shape}"
    
    comp = create_antecedent_inputs(test_data, ndays=14, window_size=14, nwindows=5)
    assert_allclose(comp.values,keras_output)
    

