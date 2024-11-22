import pandas as pd
from read_data import read_data

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers.experimental.preprocessing import Normalization, IntegerLookup, Rescaling #CategoryEncoding
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

"""Typical workflow:

1. Read the raw data in. The data should have a "case" column. datetime and case form a unique index. see read_data.read_data
   ModelBuilder: Nothing to do. Reading data is expected to be done in programatic driver code.

2. Perform any desired pre-model transformations where features become different features. Note that if you are 
   doing something like g() you may need to do it with something like:
   df['g'] = df.groupby("case").transform(["ndoi": my_gfunc])  (and make sure to use positivity preserving integration)
   ModelBuilder: 1. This is model dependent. Having transformations that are not part of TensorFlow is a consistency problem with Java, 
                 so avoid if you can. Do this outside the model
                 2. Need to initialize input_names and output_names to whatever the final names are that will be used by model. 
                 3. Need to make sure feature_dim will return the expected time dim for each feature (scalar) not including batch. For instance, if there are
                    90 lagged days for LSTM/GRU it will return 90 

3. Create lagged data (e.g. 7 individual lags and 10 11-day averaged blocks for classic CalSIM or 
   90 lagged individual for LSTM/GRU). This calcution needs to be done without crossing over cases. 
   See calc_lags_preserve_cases().
   ModelBuilder: This will usually be done by calling create_antecedent_outputs. 
                 classic CalSIM: ndays=7, window_length=11, nwindows=10
                 LSTM/GRU: could use ndays=90, window_length=0, nwindows=0
                 As noted in the Collab project, you could use the daily LSTM/GRU for everthing and embed 
                 the aggregation into TensorFlow using a convolution filter.

4. Create inputs layers for the model. This is the first TensorFlow layer that receives the input and possibly tacks on
   some normalization. The product of this step is TensorFlow architecture, not data. However the input data 
   are passed in to allow scaling to be calculated. The created layers must have one named input 
   for each conceptual feature ("sac_flow"). The 
   lags will be part of the dimensionality. The first action within the ANN may concatenate these (or you might transform a 
   subset of the variables using linear layers and then concatenate). The desired dimension is (batch=None, ntime, nfeature).
   This has caused problems in published papers.
   ModelBuilder: You may want to over-ride this with tailored scaling or eliminate scaling altogether if you have a 
                 a transformation in mind (e.g. summing Sac, SJR, exports, CU to become a surrogate for outflow)
                 and you want to defer scaling until after that so that it is farther in the model.

5. Pre-calculate the cross-validation folds for the inputs. Originally this was done by leave-1-case out but the function
   xvalid_folds() also can split cases  with a target length like 180d (actual splits will be the same length within the case
   and will be at least the target length) which keeps more data available for training. 
   The identifier of the fold will be appended to the 'fold' column. 
   ModelBuilder: As long as you have the right columns (datetime,case) and a target size for the fold (e.g. 180d) the examples should be fine.

6. Extract the output columns. Need to discuss if there is a need/ability to scale these.
   ModelBuilder: If you have set the names of columns this should work. It reindexes the output to match the input.

7. Implement build_model (for architecture) and fit_model (for individual model fit). Every fold creates a new model, which is in anticipation of 
   multithreading.   
   ModelBuilder: build_model is where you define your achitecture after the input layer. fit_model is something you build as well
                 that describes  your fitting process.

7. Pass the inputs and outputs to the xvalid(input,output) member. This function will:
   a. iterate the folds, witholding (lagged) input and aligning output for each fold forming training and test sets
   b. train the model using training. produce output or its statistics. 
   c. record the witheld output (make abstract, talk to Ryan). Standard way would be to record case 1 output for the model where case 1 is omitted. 
   d. do one more model run with nothing witheld. That will be the final model
   ModelBuilder: will try to have this be automatic if you provide the other pieces. Do this in such away that we have a full_data=True option that 
                 does not do cross-validation.

8. Save the model.

"""

def mean_squared_error2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


class ModelBuilder(object):

    def __init__(self,input_names,output_names):
        print("here we are in super()")
        self.input_names = input_names
        self.output_names = output_names
        self.ndays = -1         # Subclass should over-ride these
        self.nwindows = -1
        self.window_length = -1 
        self.ntime = 0
        




    def raw_data_to_features(self, data):
        """Converts the raw data from the model to the features named by the model
        as discoverable using  feature_names()
        """
        features = self.feature_names()
        return data[features]

    def feature_names(self):
        return self.input_names

    def feature_dim(self, feature):
        """Returns the dimension of feature, typically lags (individual or aggregated)"""
        return self.ntime

    def df_by_feature_and_time(self, df):
        """Convert a dataset with a single index with var_lag as column names and convert to MultiIndex with (var,ndx)
        This facilitates queries that select only lags or only variables. As a side effect this routine will store
        the name of the active lags for each feature, corresponding to the number of lags in the dictionary num_feature_dims)
        into the module variable lag_features.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted

        Returns
        -------
        df_var : A copy of DataFrame with multiIndex based on feature,lag  (e.g. 'sac','4d')
        """

        df = df.copy()
        indextups = []


        for col in list(df.columns):
            var = col
            lag = ""
            for key in self.feature_names():
                if key in col:
                    var = key
                    lag = col.replace(key, "").replace("_", "")
                    if lag is None or lag == "":
                        lag = "0d"
                    continue
            if var == "EC":
                lag = "0d"

            indextups.append((var, lag))

        ndx = pd.MultiIndex.from_tuples(indextups, names=("var", "lag"))
        df.columns = ndx
        names = self.feature_names()
        return df


    def prepro_layers(self, inp_layers, df):
        """Examples of processing dataframe with multiindex of locations and lags into normalized inputs"""
        layers = []
        names = self.feature_names()
        if len(names) != len(inp_layers ): 
            raise ValueError("Inconsistency in number of layers between inp_layers and feature names")
        thresh = 40000.
        dims = {x: self.feature_dim(x) for x in names} 

        for fndx, feature in enumerate(self.feature_names()):
            station_df = df.loc[:,feature]
            xinput = inp_layers[fndx]
            prepro_name=f"{feature}_prepro" 
            if feature in ["dcc", "smscg"] and False:
                feature_layer = Normalization(axis=None,name=prepro_name)  # Rescaling(1.0)
            elif feature == "sac_flow" and thresh is not None:
                feature_layer = Rescaling(1 / thresh, name=prepro_name)  # Normalization(axis=None)
            elif feature == "sjr_flow" and thresh is not None:
                feature_layer = Rescaling(0.25 / thresh, name=prepro_name)  # Normalization(axis=None)                
            else:
                feature_layer = Normalization(axis=None,name=prepro_name)
                feature_layer.adapt(
                    station_df.to_numpy())
            layers.append(feature_layer(xinput))
        return layers

    def input_layers(self):
        """Takes a dataframe representing variables 
        and processes into a list of input layers. 
        The input should be the original data without lags normalization
        should be applied if you want normalization to be applied before
        the first ANN computational layer. The example implementation shows some ways to do it.
        This would likely be overridden for different choices, though it works for many
        """
        layers = []
        names = self.feature_names()  

        for fndx, feature in enumerate(self.feature_names()):
            xinput = Input(shape=(self.feature_dim(feature)),name=feature)
            layers.append(xinput)
        
        return layers

    def build_model(self,input_layers, input_data):
        """ Builds out the architecture after input_layers through the compile. 
           The input_layers are created outside the model to ensure they follow the requested form
           Often the next stage will be to call prepro_layers as in the recursive example, 
           though this is not required
        """
        raise NotImplementedError("Must be implemented")

    
    def fit_model(self, model, fit_in, fit_out, test_in, test_out):
        """ Fits the built model 
        """
        raise NotImplementedError("Must be implemented")
    

    def calc_antecedent_preserve_cases(self, df, ndays=-1, nwindows=-1, window_length=-1):
        """
        Calculates lags for data with multiple cases so that dates alone may not be unique and lags can't be calculated naively by shifting rows
        Input df must have a "case" columns
        """

        if not "case" in df.columns:
            raise ValueError("No cases column")
        
        if ndays<0: 
            ndays = self.ndays
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length


        antecedes = []
        for case in df.case.unique():
            df_case = df.loc[df.case == case]
            
            antecedent = self.create_antecedent_inputs(df_case,ndays,window_length,nwindows)
            antecedes.append(antecedent)

        return pd.concat(antecedes,axis=0)

    def _time_fold(self, sub_df,dt):
        """ Calculates the grouping in time for a single case based on target length in time"""
        sub_df = sub_df.copy()
        start = sub_df.datetime.min()
        end = sub_df.datetime.max()
        nbin = pd.to_timedelta(end-start)//pd.to_timedelta(dt)
        sub_df['index_bounds'] = pd.cut(sub_df.index,bins=nbin,labels=False)
        return sub_df.index_bounds

    def xvalid_time_folds(self, data, target_fold_len='180d',split_in_out=True):
        """ Calculates cross-validation folds for datframe data that assume there are cases and that each 
            case is divided approximately into chunks of length target_fold_len

            Parameters
            ----------

            data : pd.DataFrame
                An incoming dataframe that has cases and dates. Within a case, dates are unique and monotone

            target_fold_len : str 
                A time delta or string that can be parsed to one
        """
        data = data.copy()
        if isinstance(target_fold_len,str):
            dt = pd.to_timedelta(target_fold_len)

        #data = data.reset_index(drop=False)  # datetime and case are columns
        data['time_fold'] = -1
        if "case" not in data:
            data["case"] = 1

        for case in data.case.unique():
    
            case_df = data.loc[data.case == case]
            case_df.loc[:,'time_fold'] =self._time_fold(case_df,dt)
            data.update(case_df)

        data['fold'] = data.groupby(['case','time_fold']).ngroup()
        
        if split_in_out:
            df_in = data[["datetime","case","fold"]+self.input_names]      # isolate input
            df_out = data[["datetime","case","fold"]+self.output_names]    # isolate output
            return df_in,df_out
        else:
            return data





    def create_antecedent_inputs(self, df, ndays=-1, window_length=-1, nwindows=-1, reverse=False):
        """
        Expands a dataframe to include lagged data.
        Each column of the input dataframe is expanded to:
        * ndays individual lagged days,  (current day + (ndays-1) previous days)
        * nwindows values of averaged data over non-overlapping blocks of window_length days per block
        The lags are ordered from most recent to farthest past, and the averaged blocks are from the days that precede the individual.

        Classic CalSIM is ndays=8, window_length=14, nwindows=5

        Returns
        -------

            df_x : pd.DataFrame
            A dataframe with input columns labeled like orig_lag1 and orig_ave1 etc for the
            individual and aggregated

        """
        
        if ndays < 0: 
            ndays = self.ndays
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length

        preserve_cols = ["datetime", "case", "fold"] if "datetime" in df.columns else ["case", "fold"]
        df2 = df[preserve_cols].copy()
        df = df.drop(preserve_cols, axis=1)
        orig_index = df.index

        if not reverse and nwindows > 0:
            raise NotImplementedError("Not Implemented.")

        if reverse:
            arr1 = [df.shift(n) for n in range(ndays)]
        else:
            arr1 = [df.shift(n) for n in reversed(range(ndays))]

        if nwindows > 0:
            dfr = df.rolling(window=window_length, min_periods=window_length).mean()
            arr2 = [dfr.shift(window_length * n + ndays) for n in range(nwindows)]                    
        else:
            arr2 = []

        df_x = pd.concat(arr1 + arr2, axis=1).dropna()  # nsamples, nfeatures


        # Adjust column names
        new_columns = [] #preserve_cols

        for n in range(ndays):
            for col in df.columns:
                if col not in preserve_cols:
                    new_columns.append(col + '_lag{}'.format(n))
        for n in range(nwindows):
            for col in df.columns:
                if col not in preserve_cols: 
                    new_columns.append(col + '_avg{}'.format(n))

        df_x.columns = new_columns
        
        df_x = df2.join(df_x, how="right")
        return df_x



#########################################################



class GRUBuilder(ModelBuilder):

    def __init__(self,input_names,output_names,ndays):
        super().__init__(input_names,output_names)
        self.ntime = ndays
        self.ndays = ndays
        self.nwindows = 0
        self.window_length = 0

    def build_model(self,input_layers, input_data):

        prepro_layers = self.prepro_layers(input_layers,input_data)          
        x = layers.Lambda(lambda x: tf.stack(x,axis=-1))(prepro_layers) 

        x = layers.GRU(units=8, return_sequences=True, #dropout=0.2, recurrent_dropout=0.2,
                            activation='sigmoid')(x)
        x = layers.GRU(units=12, return_sequences=False, #dropout=0.2, recurrent_dropout=0.2,
                            activation='sigmoid')(x)
        x = layers.Flatten()(x)
        
        outdim = len(self.output_names)
        # The regularization is unknown
        outputs = layers.Dense(units=outdim, name = "ec", activation='relu',kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(x)
        ann = Model(inputs = input_layers, outputs = outputs)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008), 
            loss='mae',   # could be mean_absolute_error or mean_squared_error 
            metrics=['mean_absolute_error','mse'],
            run_eagerly=True
        )
        return ann


    def fit_model(self,ann,fit_input,fit_output,test_input,test_output,nepochs=200):  # ,tcb):
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=nepochs,
            batch_size=32,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
            )
        return history, ann

###########################################################################################


class MLPBuilder(ModelBuilder):
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
        
        # First hidden layer 
        x = Dense(units=6, activation='sigmoid', input_dim=x.shape[1], 
                  kernel_initializer="he_normal",
                  kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden1")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Second hidden layer 
        x = Dense(units=8, activation='sigmoid', kernel_initializer="he_normal",
                  kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden2")(x) 
        x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)

        # Output layer with 1 neuron
        output = Dense(units=outdim,name="ec",activation="relu")(x)
        ann = Model(inputs = input_layers, outputs = output)

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002), 
            loss="mae", 
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