import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, backend as K
import tensorflow as tf
from numpy.testing import assert_allclose
import pytest  # Optional, but recommended for pytest usage

def stack_inputs(list_of_inputs,dim_order="lstm"):
    """ Concatenates list of individual features (sac, exports, ...) to LSTM dim conventions
        the resulting data will have dimension (nbatch, ntime, nfeature).
        Args:
             list_of_inputs : list[tf.keras.Layer]
        Returns:
             concatenated layers
    
    """
    stacked = tf.stack(outputs, axis=-1)  # Concatenate along axis=1
    return stacked
    

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



# Define the custom Conv1D layer with fixed moving average weights
class FixedConv1D(layers.Layer):
    def __init__(self, window_size, nfeatures=1, **kwargs):
        super(FixedConv1D, self).__init__(**kwargs)
        self.window_size = window_size
        self.nfeatures = nfeatures
        self.conv_layer = layers.Conv1D(filters=1, kernel_size=self.window_size, strides=self.window_size, 
                                        padding='valid', use_bias=False)
        
    def build(self, input_shape):
        # Create the fixed moving average kernel
        avg_kernel = np.ones((self.window_size, self.nfeatures, 1)) / self.window_size
        self.conv_layer.build(input_shape)
        self.conv_layer.set_weights([avg_kernel])
        self.conv_layer.trainable = False  # Make sure the layer is non-trainable
    
    def call(self, inputs):
        return self.conv_layer(inputs)



# Function to create a Conv1D layer with a fixed averaging kernel
def fixed_conv1d(input_tensor, window_size, nfeatures):
    # Create a fixed kernel for the moving average
    avg_kernel = np.ones((window_size, nfeatures, 1)) / window_size
    
    # Define a Conv1D layer and set its weights to the fixed kernel
    conv_layer = layers.Conv1D(filters=1, kernel_size=window_size, strides=window_size, padding='valid', use_bias=False)(input_tensor)
    
    # Manually set the weights of the Conv1D layer
    conv_layer.set_weights([avg_kernel])
    
    return conv_layer


# Updated function to create a Conv1D layer with a fixed moving average, reversing input and output
def create_antecedent_layer_with_conv(input_layer, ndays=14, window_size=14, nwindows=5, nfeatures=1):
    """
    Creates an antecedent layer using Conv1D to simulate rolling window aggregation,
    with a fixed moving average kernel. This approach reverses the input to process data in 
    reverse chronological order and reverses it back after applying the convolution.
    
    Args:
    input_layer : tf.keras.layers.Input
        Input layer representing the time series data for a single feature.
    ndays : int
        Number of most recent days to retain.
    window_size : int
        Window size for rolling aggregation.
    nwindows : int
        Number of windows to aggregate.
    nfeatures : int
        Number of features in the input data.
    
    Returns:
    A transformed keras layer with both individual days and rolling window aggregates.
    """
    # Ensure the input is reshaped correctly
    reshaped_input = layers.Reshape((input_layer.shape[1], nfeatures))(input_layer)
    
    # Step 1: Retain the most recent `ndays` (index 0 to `ndays-1`)
    most_recent_ndays = layers.Lambda(lambda x: x[:, :ndays, :])(reshaped_input)  # Slice the most recent ndays
    
    # Step 2: Slice the input starting from `index=ndays` for rolling aggregation
    rolling_input = layers.Lambda(lambda x: x[:, ndays:, :])(reshaped_input)  # Start rolling from index ndays

    # Step 3: Reverse the input to apply the kernel in reverse chronological order
    reversed_input = layers.Lambda(lambda x: K.reverse(x, axes=1))(rolling_input)
    
    # Step 4: Apply the custom FixedConv1D layer for rolling window means
    rolling_means = FixedConv1D(window_size, nfeatures)(reversed_input)
    
    # Step 5: Reverse the output to restore the original order
    reversed_output = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))(rolling_means)
    
    # Step 6: Concatenate the most recent days and rolling means to form the final feature set
    aggregated_features = layers.Concatenate(axis=1)([most_recent_ndays, reversed_output])

    # Assuming `aggregated_features` has the shape (batch_size, time_steps, nfeatures)
    # Use Reshape to remove the last dimension
    aggregated_features = layers.Reshape((aggregated_features.shape[1],))(aggregated_features)
    
    return aggregated_features