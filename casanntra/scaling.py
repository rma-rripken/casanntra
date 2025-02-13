from tensorflow.keras.layers import Layer
import tensorflow as tf


class ModifiedExponentialDecayLayer(Layer):
    """
    A Keras layer that applies a modified exponential decay transformation.

    This layer transforms input values using an exponential decay function,
    with tapering applied to ensure a well-bounded range.

    Parameters
    ----------
    a : float, optional
        Decay rate (default is `1e-5`). Controls the rate of exponential decay.
    b : float, optional
        Upper bound for tapering (default is `70000`). Prevents extreme decay values.
    **kwargs : dict
        Additional keyword arguments for the `Layer` base class.

    Attributes
    ----------
    a : float
        Decay rate parameter.
    b : float
        Tapering threshold.

    Methods
    -------
    call(inputs)
        Applies the modified exponential decay transformation.
    get_config()
        Returns the layer's configuration for serialization.

    Returns
    -------
    tf.Tensor
        Transformed tensor with the same shape as the input.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from scaling import ModifiedExponentialDecayLayer
    >>> layer = ModifiedExponentialDecayLayer(a=1e-5, b=70000)
    >>> inputs = tf.constant([1000.0, 50000.0, 100000.0])
    >>> outputs = layer(inputs)
    >>> print(outputs.numpy())

    Notes
    -----
    - The transformation is defined as:
      
      .. math::
         f(x) = \\frac{e^{-ax} - e^{-ab}}{1 - e^{-ab}}

    - The denominator ensures that the function is well-bounded.
    - The parameters `a` and `b` are **fixed** (not trainable).
    """

    def __init__(self, a=1e-5, b=70000, **kwargs):
        super(ModifiedExponentialDecayLayer, self).__init__(**kwargs)
        self.a = a
        self.b = b

    def call(self, inputs):
        # Implement the modified exponential decay function
        exp_decay = tf.exp(-self.a * inputs)
        exp_decay_tapered = (exp_decay - tf.exp(-self.a * self.b)) / (
            1 - tf.exp(-self.a * self.b)
        )
        return exp_decay_tapered

    def get_config(self):
        # Return the configuration of the layer for serialization
        config = super(ModifiedExponentialDecayLayer, self).get_config()
        config.update(
            {
                "a": self.a,
                "b": self.b,
            }
        )
        return config


class TunableModifiedExponentialDecayLayer(Layer):
    """
    A Keras layer that applies a modified exponential decay transformation with trainable parameters.

    This layer is similar to `ModifiedExponentialDecayLayer`, but allows the decay rate (`a`)
    and tapering threshold (`b`) to be **learned** during training.

    Parameters
    ----------
    a : float, optional
        Initial decay rate (default is `1e-5`). Controls the initial rate of exponential decay.
    b : float, optional
        Initial upper bound for tapering (default is `70000`).
    **kwargs : dict
        Additional keyword arguments for the `Layer` base class.

    Attributes
    ----------
    a : tf.Variable
        Trainable decay rate.
    b : tf.Variable
        Trainable tapering threshold.

    Methods
    -------
    build(input_shape)
        Initializes trainable weights for `a` and `b`.
    call(inputs)
        Applies the modified exponential decay transformation.
    get_config()
        Returns the layer's configuration for serialization.

    Returns
    -------
    tf.Tensor
        Transformed tensor with the same shape as the input.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from scaling import TunableModifiedExponentialDecayLayer
    >>> layer = TunableModifiedExponentialDecayLayer(a=1e-5, b=70000)
    >>> inputs = tf.constant([1000.0, 50000.0, 100000.0])
    >>> outputs = layer(inputs)
    >>> print(outputs.numpy())

    Notes
    -----
    - **This layer is untested.** Use with caution.
    - The transformation formula is the same as `ModifiedExponentialDecayLayer`, but `a` and `b` are **trainable**.
    - The denominator in the transformation is clipped to avoid division by zero.
    - Debugging print statements are included in `build()`.

    """    
    def __init__(self, a=1e-5, b=70000, **kwargs):
        super(TunableModifiedExponentialDecayLayer, self).__init__(**kwargs)
        self.initial_a = a
        self.initial_b = b

    def build(self, input_shape):
        self.a = self.add_weight(
            name="a",
            shape=(),
            initializer=tf.constant_initializer(self.initial_a),  # Explicit initializer
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(),
            initializer=tf.constant_initializer(self.initial_b),  # Explicit initializer
            trainable=True,
        )
        print(f"Initialized a: {self.a.numpy()}, b: {self.b.numpy()}")  # Debug print
        super(TunableModifiedExponentialDecayLayer, self).build(input_shape)

    def call(self, inputs):
        exp_decay = tf.exp(-self.a * inputs)
        clipped_denominator = tf.clip_by_value(
            1 - tf.exp(-self.a * self.b), 1e-6, float("inf")
        )
        exp_decay_tapered = (exp_decay - tf.exp(-self.a * self.b)) / clipped_denominator
        return exp_decay_tapered

    def get_config(self):
        config = super(TunableModifiedExponentialDecayLayer, self).get_config()
        config.update(
            {
                "a": self.initial_a,
                "b": self.initial_b,
            }
        )
        return config


custom_scaling = {"ModifiedExponentialDecayLayer": ModifiedExponentialDecayLayer}
