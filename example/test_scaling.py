""" Informal testbed for scaling funcitons, mostly intended for Sac/NDO/Northern"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define individual functions
#def modified_exponential_decay(x, a=5e-6, b=40000):  #SJR
def modified_exponential_decay(x, a=5e-7, b=70000): #SAC/NDO
    exp_decay = tf.exp(-a * x)
    exp_decay_tapered = (exp_decay - tf.exp(-a * b)) / (1 - tf.exp(-a * b))
    return exp_decay_tapered

def piecewise_decay(x, c=40000, b=70000, a=1e-1):
    linear_part = tf.where(x <= c, 1 - x / c, 0)
    exponential_part = tf.where(x > c, (1 - c / b) * tf.exp(-a * (x - c)), 0)
    return linear_part + exponential_part

def logistic_compression(x, a=1e-4, c=40000):
    return 1 / (1 + tf.exp(a * (x - c)))

def custom_compression(x, scale=40000):
    return 1 / (1 + x / scale)

# Define the factory
def function_factory(name, **kwargs):
    """
    Factory to choose and configure a decay function.

    Parameters:
        name (str): Name of the function to use. Options: 'exponential', 'piecewise', 'logistic', 'custom'.
        kwargs: Parameters to configure the chosen function.

    Returns:
        A callable function configured with the specified parameters.
    """
    if name == 'exponential':
        return lambda x: modified_exponential_decay(x, **kwargs)
    elif name == 'piecewise':
        return lambda x: piecewise_decay(x, **kwargs)
    elif name == 'logistic':
        return lambda x: logistic_compression(x, **kwargs)
    elif name == 'custom':
        return lambda x: custom_compression(x, **kwargs)
    else:
        raise ValueError(f"Unknown function name: {name}. Choose from 'exponential', 'piecewise', 'logistic', 'custom'.")

if __name__ == "__main__":
    # Example usage
    chosen_function = function_factory('exponential')
    xval = np.arange(0.,200000.,5000.)
    x = tf.constant(xval, dtype=tf.float32)
    result = chosen_function(x)
    print(result.numpy())
    plt.plot(xval,result.numpy())
    plt.show()
