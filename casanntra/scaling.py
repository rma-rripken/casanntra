from tensorflow.keras.layers import Layer
import tensorflow as tf

class ModifiedExponentialDecayLayer(Layer):
    def __init__(self, a=1e-5, b=70000, **kwargs):
        super(ModifiedExponentialDecayLayer, self).__init__(**kwargs)
        self.a = a
        self.b = b

    def call(self, inputs):
        # Implement the modified exponential decay function
        exp_decay = tf.exp(-self.a * inputs)
        exp_decay_tapered = (exp_decay - tf.exp(-self.a * self.b)) / (1 - tf.exp(-self.a * self.b))
        return exp_decay_tapered

    def get_config(self):
        # Return the configuration of the layer for serialization
        config = super(ModifiedExponentialDecayLayer, self).get_config()
        config.update({
            "a": self.a,
            "b": self.b,
        })
        return config

class TunableModifiedExponentialDecayLayer(Layer):
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
        clipped_denominator = tf.clip_by_value(1 - tf.exp(-self.a * self.b), 1e-6, float("inf"))
        exp_decay_tapered = (exp_decay - tf.exp(-self.a * self.b)) / clipped_denominator
        return exp_decay_tapered

    def get_config(self):
        config = super(TunableModifiedExponentialDecayLayer, self).get_config()
        config.update({
            "a": self.initial_a,
            "b": self.initial_b,
        })
        return config