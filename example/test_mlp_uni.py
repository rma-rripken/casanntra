from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit




class MLPUniBuilder(ModelBuilder):
    """ Example ModelBuilder using CalSim-like MLP. """

    def __init__(self,input_names,output_names,ndays,nwindows,window_length):
        super().__init__(input_names,output_names)
        print("Setting params")
        self.ndays = ndays
        self.nwindows = nwindows
        self.window_length = window_length
        self.ntime = ndays + nwindows

    def build_model(self,input_layers, input_data):
        """ Builds the standard CalSIM ANN
            Parameters
            ----------
            input_layers : list  
            List of tf Input layers

            input_data: dataframe
            Full list of data, without lags at this point. This is used for calling prepro_layers for scaling.
        """        
        outdim = len(self.output_names)
        prepro_layers = self.prepro_layers(input_layers,input_data)   

        x = layers.Concatenate()(prepro_layers) 
        
        # First hidden layer with 8 neurons and sigmoid activation function
        x = Dense(units=4, activation='sigmoid', input_dim=x.shape[1], 
                  kernel_initializer="he_normal",
                  kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden1")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Second hidden layer 
        x = Dense(units=2, activation='sigmoid', kernel_initializer="he_normal",
                  kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden2")(x) 
        x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)

        # Output layer with 1 neuron
        output = Dense(units=outdim,name="ec",activation="relu")(x)
        ann = Model(inputs = input_layers, outputs = output)

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002), 
            loss="mse", 
            metrics=['mean_absolute_error'],
            run_eagerly=True
        )

        return ann

    def fit_model(self,ann,fit_input,fit_output,test_input,test_output,nepochs=200):
        tf.config.run_functions_eagerly(True)
        #tf.data.experimental.enable_debug_mode()
        history = ann.fit(
                fit_input,fit_output, 
                validation_data=(test_input, test_output), 
                batch_size=32, 
                epochs=nepochs, 
                verbose=2,
                shuffle=True
                )
        
        return history, ann




def test_xvalid_mlp():
    #"sf_tidal_filter",
    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy",
                   "sf_tidal_filter","dcc","smscg"]
    output_names = ["cse"]
    plot_locs = ["cse"]

    builder = MLPUniBuilder(input_names=input_names,
                     output_names=output_names,
                     ndays=14,nwindows=4,window_length=14)

    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)
    df_in, df_out = builder.xvalid_time_folds(df,target_fold_len='180d',split_in_out=True)   # adds a column called 'fold'


    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:,col] = df_out.loc[:,col]/100. 
        elif col in ["mal","cse","bdl","nsl2"]:
            df_out.loc[:,col] = df_out.loc[:,col]/10000.
        elif col in ["emm2","jer","tms"]:
            df_out.loc[:,col] = df_out.loc[:,col]/2000.
        else:
            df_out.loc[:,col] = df_out.loc[:,col]/1000.


    #xvalid_fit(df_in,df_out,builder,plot_folds=[0,1],plot_locs=plot_locs)
    xvalid_fit(df_in,df_out,builder,plot_folds="all",plot_locs=plot_locs,out_prefix="output/mlp_uni")



if __name__ == "__main__":
    test_xvalid_mlp()