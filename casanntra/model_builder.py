import pandas as pd
from casanntra.read_data import read_data
from casanntra.scaling import ModifiedExponentialDecayLayer,TunableModifiedExponentialDecayLayer
from keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.layers.experimental.preprocessing import Normalization, IntegerLookup, Rescaling #CategoryEncoding
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os


"""Typical workflow. The library does this:

1. Read the raw data in. The data should have a "case" column. datetime and case form a unique index. see read_data.read_data
   ModelBuilder: Nothing to do. Reading data is expected to be done in programatic driver code.

2. Perform any desired pre-model transformations. Features remain individual inputs, typically, to be concatenated in build_model.
   
   There are a lot of notes on the subject of transformations and scaling belowHaving transformations that are not part of TensorFlow is a consistency problem with Java, 
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

4. Create inputs layers for the model in prepro_layers. 
   This is the first TensorFlow layer that receives the input and is often modified to tack on
   some normalization. The product of this step is TensorFlow layers, not data. However the input data 
   are passed in to allow scaling to be calculated or adapted. The created layers must have one named input 
   for each conceptual feature ("sac_flow").  Lags will be part of the dimensionality of the feature. 
   
   The first action within the ANN may concatenate these (or you might transform a 
   subset of the variables using linear layers and then concatenate). The desired dimension is (batch=None, ntime, nfeature).
   
   The docstring for base class of prepro_layers() has considerable discussion about scaling and how the challenges
   of DSM2 and SCHISM/RMA differe and pointing out that what what you purely for DSM2 probably will reduce the potential for the other two 

5. Pre-calculate the cross-validation folds for the inputs. Originally this was done by leave-1-case out but the function
   xvalid_folds() also can split cases  with a target length like 180d (actual splits will be the same length within the case
   and will be at least the target length) which keeps more data available for training. 
   
   In fit_from_config.py, you will also find an option to reduce the number of folds to the number of processors. That will be machine
   dependent and we (DWR) tend to have big machines, so should re-implement to make specifying a number feasible.

   The identifier of the fold is integer that will be appended to the 'fold' column. The timing of this is after the lags have been
   calculated 
   ModelBuilder: As long as you have the right columns (datetime,case) and a target size for the fold (e.g. 180d) the examples should be fine.

6. Extract the output columns. Need to discuss if there is a need/ability to scale these. In the examples it is done separately and that won't 
   work long term. This is currently done in the implementations and I do the scaling based on the D1641 critical year maximum objective. 

7. Implement build_model (for architecture) and fit_model (for individual model fit). Every fold creates a new copy of the model, 
   which is needed for safe multithreading.   
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

class UnscaleLayer(tf.keras.layers.Layer):
    def __init__(self, output_scales, **kwargs):
        super(UnscaleLayer, self).__init__(**kwargs)
        self.output_scales = tf.constant(output_scales, dtype=tf.float32)

    def call(self, inputs):
        return inputs * self.output_scales  # ✅ Rescales ANN’s output

    def get_config(self):
        config = super().get_config()
        config.update({"output_scales": self.output_scales.numpy().tolist()})
        return config


def mean_squared_error2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


class ModelBuilder(object):
    def __init__(self, input_names, output_names, ndays=90):
        """Base class for model builders, now handling custom object registration."""
        self.input_names = input_names
        self.output_names = output_names  # Dict of {output_name: scale_factor}
        self.ndays = ndays
        self.load_model_fname = None  # Used for transfer learning

        # ✅ Centralized custom object registration
        self.custom_objects = {"UnscaleLayer": UnscaleLayer}

    def register_custom_object(self, name, obj):
        """Allows subclasses to register additional custom objects."""
        self.custom_objects[name] = obj

    def load_existing_model(self):
        """Handles model loading and ensures all required custom objects are registered."""
        if self.load_model_fname is None:
            return None  # No model to load
        
        print(f"Loading model from {self.load_model_fname} with registered custom objects.")
        base_model = load_model(self.load_model_fname, custom_objects=self.custom_objects)
        base_model.load_weights(self.load_model_fname.replace(".h5", ".weights.h5"))

        return base_model


    def _create_unscaled_layer(self, scaled_output):
        """Creates an unscaled version of an existing output layer for inference purposes."""
        output_scales = list(self.output_names.values())  # Extract scale factors

        # ✅ Unscaled output for inference
        return UnscaleLayer(output_scales, name="unscaled_output")(scaled_output)



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
        """ Create the preprocessing layers, one per location, which will be concatenated later.
            This function performs any sort of special scaling. Here the superclass is overridden.

            Scaling for Bay-Delta problems is inherantly custom. Prior work has proceeded on the 
            hope that one can use a "best practice" like Normalize (mean and std). These are 
            questionably appropriate because the flow variables saturate an order of magnitude lower than
            their upper limits. (e.g. above 40,000cfs for Sac). 

            For DSM2, which can't model low X2, the challenge is to try to saturate at high outflow and 
            scale the inflows so that the salinity-affecting range is some reasonable (0-1 ish) range.
             
            Scaling rivers with volatile statistics does not achieve this. For instance, Sac River 
            flow maximum depends wildly on the choice of year, and if you are doing transfer learning
            from one group of years to another this is erratic.  The range could be as much as 300,000 
            which compresses the portion pertinent for salinity management into Q' < 0.1. Above this 
            number, DSM2 doesn't have any variables that are sensitive to the inputs ... so in a sense
            you might even be better off omitting these data since they don't help the solve.

            One of the original solutions to this was to just use a threshold like 40,000cfs for the Sacramento River 
            and 10,000cfs for the SJR. It works well in DSM2. The required saturation above 70kcfs seems to happen
            by good fortune. 
             
            When you anticipate transferring from DSM2 to SCHISM/RMA you might plan for the big difference in
            the ability of those to models to reproduce X2 < 55km, which means that they have meaningful 
            gradients for training under much larger flows than DSM2 does. In fact they might (?) be hobbled 
            by a 40kcfs saturation. An example custom function based on modified exponential decay is 
            available in scaling.py and an example of its use may be found, sometiems commented, in fit_from_config.py. 
    
        """
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
            elif feature in [ "sac_flow", "ndo"] and thresh is not None:
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
    
    def register_custom_object(self, name, obj):
        """Allows subclasses to register additional custom objects."""
        self.custom_objects[name] = obj    

    def build_model(self, input_layers, input_data, add_unscaled_output=False):
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
    

    def calc_antecedent_preserve_cases(self, df, ndays=-1, nwindows=-1, window_length=-1,reverse=None):
        """
        Calculates lags for data with multiple cases so that dates alone may not be unique and lags can't be calculated naively by shifting rows
        Input df must have a "case" columns
        
        Parameters
        ----------
        df : DataFrame
            A dataframe that has a 'datetime' and 'cases' column and for which datetime,cases are a unique key
        
        ndays : int
            Number of individual day lags

        nwindows: int
            Number of aggregated windows of days, which will start preceding ndays
        
        window_length : int
            Duration of days of the windows

        reverse : bool
            Whether to return the lags in backward looking order starting with "now". The default for LSTM is no, 
        since this is part of the LSTM/GRU architecture. For the CalSim MLPs this is customarily True. The method
        currently raises an error if nwindows > 0 and reverse=False because this isn't plumbed out
        """

        if not "case" in df.columns:
            raise ValueError("No cases column")
        
        if reverse is None:
            reverse = self.reverse_time_inputs

        if not reverse and nwindows>0:
            raise NotImplementedError("Not implmented for non-reverse plus aggregation windows.")

        if ndays<0: 
            ndays = self.ndays
        if nwindows < 0:
            nwindows = self.nwindows
        if window_length < 0:
            window_length = self.window_length


        antecedes = []
        for case in df.case.unique():
            df_case = df.loc[df.case == case]
            
            antecedent = self.create_antecedent_inputs(df_case,ndays,window_length,nwindows,reverse)
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

    def xvalid_time_folds(self, data, target_fold_len='180d',split_in_out=True, reverse=False):
        """ Calculates cross-validation folds for datframe data that assume there are cases and that each 
            case is divided approximately into chunks of length target_fold_len

            Parameters
            ----------

            data : pd.DataFrame
                An incoming dataframe that has cases and dates. Within a case, dates are unique and monotone

            target_fold_len: pandas freq like '180d'
                A length, typically smaller than the cases, that will be used to create smaller xvalidation folds. For
            instance, the cases might be up to 2 years long and target_fold_len='180d'.

            split_in_out: bold
                Whether to return input and output separately. Helps reduce bookkeeping

            reverse: bool 
                Whether to return lags in reverse chronicalogical order, something the CalSim models do 

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
            df_out = data[["datetime","case","fold"]+list(self.output_names)]    # isolate output
            return df_in,df_out
        else:
            return data





    def create_antecedent_inputs(self, df, ndays=-1, window_length=-1, nwindows=-1, reverse=None):
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
        if reverse is None: reverse = self.reverse_time_inputs

        preserve_cols = [x for x in ["datetime", "case", "fold"] if x in df.columns]
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

class GRUBuilder2(ModelBuilder):
    """
    This model builder is the current (2025-01-20) standard recursive model
    """

    def __init__(self,input_names,output_names,ndays,load_model_fname=None):
        super().__init__(input_names,output_names,load_model_fname)
        self.ntime = ndays                  
        self.ndays = ndays                  # Number of individual days
        self.nwindows = 0                   # Number of windows, always zero for recursive
        self.window_length = 0              # Length of each windows
        self.reverse_time_inputs = False    # Reorder days looking backward. 
                                            # False for recursive, True (by convention) for 
        self.register_custom_object("ModifiedExponentialDecayLayer", ModifiedExponentialDecayLayer)
        self.register_custom_object("TunableModifiedExponentialDecayLayer", TunableModifiedExponentialDecayLayer)
        
    def prepro_layers(self, inp_layers, df):
        """ Create the preprocessing layers, one per location, which will be concatenated later.
            This function performs any sort of special scaling. Here the superclass is overridden.
            See the base class comments in bodel_builder.py for discussion. 
        """
        if df is None:
            raise ValueError("Invalid (None) data frame.")
        
        layers = []
        names = self.feature_names()
        if len(names) != len(inp_layers ): 
            raise ValueError("Inconsistency in number of layers between inp_layers and feature names")
        thresh = 40000.
        dims = {x: self.feature_dim(x) for x in names} 

        for fndx, feature in enumerate(self.feature_names()):
            if not feature in df.columns: 
                raise ValueError(f"Feature not found in dataframe: {feature}")
            station_df = df.loc[:,feature]
            xinput = inp_layers[fndx]
            prepro_name=f"{feature}_prepro" 
            if feature in ["dcc", "smscg"] and False:
                feature_layer = Normalization(axis=None,name=prepro_name)  # Rescaling(1.0)
            elif feature in [ "sac_flow", "ndo"] and thresh is not None:
                # Define the model. Use the test_scaling file to refine parameters
                feature_layer = ModifiedExponentialDecayLayer(a=1.e-5, b=70000., name=prepro_name)
                #feature_layer = Rescaling(1 / thresh, name=prepro_name)  # Normalization(axis=None)
            elif feature == "sjr_flow" and thresh is not None:
                feature_layer = ModifiedExponentialDecayLayer(a=1.e-5, b=40000., name=prepro_name)
                #feature_layer = Rescaling(0.25 / thresh, name=prepro_name)  # Normalization(axis=None)
            elif feature == "exports":
                feature_layer = Rescaling(0.0001, name=prepro_name)
            elif feature == "delta_cu":
                freature_layer = Rescaling(1/3000.,name=prepro_name)                
            else:
                feature_layer = Normalization(axis=None,name=prepro_name)
                feature_layer.adapt(
                    station_df.to_numpy())
            layers.append(feature_layer(xinput))
        return layers



    def build_model(self,input_layers, input_data):
        """ Build or load the model architecture. Or load an old one and extend."""
        
        do_load = self.load_model_fname is not None
        print(f"do_load={do_load}, load_model_fname={self.load_model_fname}")
        if do_load:
            print(f"Loading from {self.load_model_fname} for refinement")
            ann = load_model(self.load_model_fname,custom_objects=self.custom_objects)
            ann.load_weights(self.load_model_fname.replace(".h5",".weights.h5"))

            print(ann.summary())
            return ann
        else:
            print(f"Creating from scratch")
            prepro_layers = self.prepro_layers(input_layers,input_data)          
            x = layers.Lambda(lambda x: tf.stack(x,axis=-1))(prepro_layers) 
            x = layers.GRU(units=32, return_sequences=True, 
                                activation='sigmoid',name='gru_1')(x)
            x = layers.GRU(units=16, return_sequences=False, 
                                activation='sigmoid',name='gru_2')(x)
            x = layers.Flatten()(x)
            
            outdim = len(self.output_names)
            # The regularization is unknown
            outputs = layers.Dense(units=outdim, name = "ec", activation='elu',
                                kernel_regularizer = regularizers.l1_l2(l1=0.0001,l2=0.0001))(x)
            ann = Model(inputs = input_layers, outputs = outputs)
            print(ann.summary())

        return ann


    def fit_model(self,
                ann,
                fit_input,
                fit_output,
                test_input,
                test_output,
                init_train_rate,
                init_epochs,
                main_train_rate,
                main_epochs):  
        """Custom fit_model for MultiStageModelBuilder to support staged learning."""

        # ✅ Select the correct loss function
        if self.transfer_type == "mtl":
            loss_function = weighted_mtl_loss
        elif self.transfer_type == "contrastive":
            loss_function = contrastive_loss
        else:
            loss_function = "mae"  # Default for DSM2 base

        print(f"Using loss function: {loss_function}")

        # ✅ Initial training phase (faster learning rate)
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=1.0), 
            loss=loss_function,
            metrics=['mae', 'mse'],
            run_eagerly=True
        )
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
        )

        # ✅ Main training phase (slower learning rate)
        if main_epochs is not None and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate), 
                loss=loss_function,
                metrics=['mae', 'mse'],
                run_eagerly=False
            )
            history = ann.fit(
                fit_input,
                fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True
            )

        return history, ann


###########################################################################################

class MLPBuilder1(ModelBuilder):
    """ Example ModelBuilder using CalSim-like MLP. """

    def __init__(self,input_names,output_names,ndays,nwindows,window_length):
        super().__init__(input_names,output_names,ndays,nwindows,window_length)


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
        x = Dense(units=12, activation='linear', input_dim=x.shape[1], 
                  kernel_initializer="he_normal",
                  #kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden1")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Second hidden layer 
        x = Dense(units=8, activation='sigmoid', kernel_initializer="he_normal",
                  #kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.00001),
                  name="hidden2")(x) 
        x = tf.keras.layers.BatchNormalization(name="batch_normalize")(x)

        # Output layer with 1 neuron per output
        output = Dense(units=outdim,name="ec",activation="relu")(x)
        ann = Model(inputs = input_layers, outputs = output)
        print(ann.summary())
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

