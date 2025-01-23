# Reduce regularization
# Larger model
# MSE or weighted MAE


from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi,bulk_fit
from casanntra.tide_transforms import *
from keras.models import load_model

from sklearn.decomposition import PCA

class GRUBuilder2m(ModelBuilder):
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
                                            # False for recursive, True (by convention) for MLP
        
    def prepro_layers(self, inp_layers, df):
        """ Create the preprocessing layers, one per location, which will be concatenated later.
            This function performs any sort of special scaling. Here the superclass is overridden.
            Many of the scales are chosen to saturate above the effective limit (e.g. above 40,000cfs for Sac).
            Examples of processing dataframe with multiindex of locations and lags into normalized inputs. Other 
            normalization ideas are contained here: 

            Do not recommend scaling rivers with volatile upper limits based on that. For instance, Sac River 
            maximum depends capriciously on the choice of year, and the range will be as much as 300,000 which compresses
            the usable range for salinity into the range [0,0.1]. Alternatively we can do transformations.
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


    def build_model(self,input_layers, input_data):
        """ Build or load the model architecture. Or load an old one and extend."""
        
        do_load = self.load_model_fname is not None
        print(f"do_load={do_load}, load_model_fname={self.load_model_fname}")
        if do_load:
            print(f"Loading from {self.load_model_fname} for refinement")
            ann = load_model(self.load_model_fname)
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
                  main_epochs):  # ,tcb):
        """ Performs the fit in two stages, one with a faster than normal learning rate and then a normal choice.
             This seems to work better than the natural Adam adaptation."""
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate), 
            loss='mae',   # could be mean_absolute_error or mean_squared_error 
            metrics=['mean_absolute_error','mse'],
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
        do_main = (main_epochs is not None) and (main_epochs>0)
        if do_main:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate), 
                loss='mae',   # could be mean_absolute_error or mean_squared_error 
                metrics=['mean_absolute_error','mse'],
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

def fit_from_config(
    name,
    input_prefix,       # prefix for input csv files (minus "_1.csv")
    output_prefix, 
    input_mask_regex,  # None     # regex to filter out (not implemented)
    save_model_fname,  # Name for saved model, e.g. dsm2_base_gru2.h5
    load_model_fname,  # Name for loading model or None if it is an original fit
    pool_size,         # Pool size 1 or 2x pool size would cover all the folds
    target_fold_length, # 180d
    pool_aggregation,   # Collapses cross-validation folds for very large # cases (e.g. 100+) to available pool. Used for DSM2 warm up
    init_train_rate,    # Training rate in warm up pass 0.008. Couls also be used as sole pass for transfer learning 
    init_epochs,        # Num of epochs for warm up pass, say 10
    main_train_rate,    # Training rate for majority of iterations
    main_epochs,        # Example :110
    pca_tides_approx=False     # Should tidal quantities be collapsed using appoximate PCA weights
):  
    """ Performsa configured sequence of named fitting steps from yaml
        The creation of the model builder is not configured yet probably not hard with some factories and lists of
        input and output stations."""

    print(f"Processing config {name}")
    # todo: note derivative
    if pca_tides_approx:
        input_names = [ "sac_flow","exports","sjr_flow","cu_flow","tidal_pc1","tidal_pc2","dcc","smscg"]
    else:
        input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","sf_tidal_filter","dcc","smscg"]    
    
    output_names = ["x2","pct","mal","god","vol","bdl","nsl2","cse","emm2","tms","anh","jer",
                    "gzl","sal","frk","bac","rsl","oh4","trp"]
    plot_locs = ["x2","cse","emm2","jer","bdl","sal","bac"]
    builder = GRUBuilder2m(input_names=input_names,output_names=output_names,ndays=80)
    builder.load_model_fname = load_model_fname

    fpattern = f"{input_prefix}_*.csv"
    df = read_data(fpattern)

    # Uses pre-calculated statistics and PCA weights so that this standardizes across transfer learning
    # No longer recommended; energy and filtered levels are sufficiently orthogonal
    if pca_tides_approx:    
        df = append_tidal_pca_cols_approx(df) 

    
     # Breaks the data into cross-validation folds 
    df_in, df_out = builder.xvalid_time_folds(df,target_fold_length,split_in_out=True)   # adds a column called 'fold'
    # The folds will be too numerous with the 100-year sets, so this re configures them around the 
    # multithreading pool size.
    if pool_aggregation:
        print(f"pool_aggregation={pool_aggregation} and pool_size={pool_size}")
        df_in['fold'] = df_in['fold'] % pool_size
        df_out['fold'] = df_out['fold'] % pool_size

    # Needs to be moved into configuration, and eventually it needs to be part of the model 
    # otherwise it represents a coordination problem.
    # I just use heuristic scaling of outputs based on known orders of magnitude
    # built around critical year values or D-1641 standards so that "1.0" is approximately there.
    # More portable and just as effective as relying on period-dependent statistics,
    # but the real solution is to encode it in a layer.
    for col in output_names:
        if col == "x2":
            df_out.loc[:,col] = df_out.loc[:,col]/100. 
        elif col in ["mrz","pct","mal","gzl","god","vol","cse","bdl","nsl2"]:
            df_out.loc[:,col] = df_out.loc[:,col]/12000.
        elif col in ["jer","frk"]:            
            df_out.loc[:,col] = df_out.loc[:,col]/2500.
        elif col in ["emm2","tms","anh"]:
            df_out.loc[:,col] = df_out.loc[:,col]/3000.            
        else:
            df_out.loc[:,col] = df_out.loc[:,col]/1500.

    # Do leave-one-fold out validation
    xvalid_fit_multi(df_in,df_out,builder,plot_folds="all",plot_locs=plot_locs,
                     out_prefix=output_prefix, 
                     init_train_rate=init_train_rate,
                     init_epochs=init_epochs, 
                     main_train_rate=main_train_rate, 
                     main_epochs=main_epochs, 
                     pool_size=pool_size)
    
    print("Doing bulk fit")
    # Cross validation does a round robin of fits, none of which is the "official" one
    # bulk_fit fits to all the data. So ideally you would use xvalid_fit_multi to characterize loss
    # and this function to create one you can save and carry forth to the next training step.
    # loss from this fit is not useful.
    ann = bulk_fit(builder,df_in,df_out,                      # consistency of where builder is
                   output_prefix,                             # todo: unused
                   fit_in=df_in,fit_out=df_out,
                   test_in=df_in,test_out=df_out,
                   init_train_rate=init_train_rate,
                   init_epochs=init_epochs, 
                   main_train_rate=main_train_rate, 
                   main_epochs=main_epochs)
    
    # Save the model out. The save_weights step addresses a bug in TF 2.7.4
    print(f"Saving model {name} to {save_model_fname}")
    ann.save_weights(save_model_fname.replace(".h5",".weights.h5"))
    ann.save(save_model_fname,overwrite=True)

def read_config(configfile):
    import yaml
    with open(configfile) as stream:
        try:
            content = yaml.safe_load(stream)
            return content
        except yaml.YAMLError as exc:
            print(exc)

def fit_step(step):
    print(step)


def process_config(configfile,proc_steps): 
    config = read_config(configfile)
    proc_all = (proc_steps == "all") 
    if proc_steps is None: 
        raise ValueError("Processing steps or the string 'all' required")
    elif not isinstance(proc_steps,list):
        if isinstance(proc_steps,str): 
            proc_steps = [proc_steps]

    for step in config['steps']:
        if step['name'] in proc_steps or proc_all:
            print("\n\n###############  STEP",step['name'],   "############\n")
            for key in step:
                if step[key] == "None":
                    step[key] = None
            fit_from_config(**step)

def main():
    configfile = "transfer_config.yml"
    process_config(configfile,["dsm2.schism"])

if __name__ == "__main__":
    main()