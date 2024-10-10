
# Functions
def mse_loss_masked(y_true, y_pred):
    squared_diff = tf.reduce_sum(tf.math.squared_difference(y_pred[y_true > 0], y_true[y_true > 0]))
    return squared_diff / (tf.reduce_sum(tf.cast(y_true > 0, tf.float32)) + 0.01)


def build_lstm_v1(inputs,outdim,units=12):

    x = inputs
    x = layers.GRU(units, return_sequences=True,
                           activation='sigmoid')
    x = layers.GRU(units, return_sequences=False,
                           activation='sigmoid')
    x = keras.layers.Flatten(x)
    # Originally this was linear, but relu is positivity preserving as long as outdim
    # outputs have been scaled (divided) but not centered (e.g. mean removed)
    # Ryan mentioned some difficulties
    x = keras.layers.Dense(outdim,name="ec",activation="relu")  

def build_calsim_mpl_v1(inputs,outdim):
    """ Builds a standard CalSIM ANN
        Parameters
        ----------
        layers : list  
        List of tf.Layers

        inputs: dataframe
    """        

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=x.shape[1], kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', kernel_initializer="he_normal",name="hidden")(x) 
    x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)
    
    # Output layer with 1 neuron
    output = Dense(units=outdim,name="ec",activation="relu")(x)
    ann = Model(inputs = inputs, outputs = output)
    

    
    return ann, tensorboard_cb