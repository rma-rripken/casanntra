# Reduce regularization
# Larger model
# MSE or weighted MAE


from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi,bulk_fit
from casanntra.tide_transforms import *
from casanntra.scaling import (ModifiedExponentialDecayLayer,
                              TunableModifiedExponentialDecayLayer)
from keras.models import load_model
from sklearn.decomposition import PCA


def fit_from_config(
    builder,
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

    #print(f"Processing config {name}")
    # todo: note derivative
    #if pca_tides_approx:
    #    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","tidal_pc1","tidal_pc2","dcc","smscg"]
    #else:
    #    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","sf_tidal_filter","dcc","smscg"]    
    
    #output_names = ["x2","pct","mal","god","vol","bdl","nsl2","cse","emm2","tms","anh","jer",
    #                "gzl","sal","frk","bac","rsl","oh4","trp"]
    #plot_locs = ["x2","cse","emm2","jer","bdl","sal","bac"]
    #builder = GRUBuilder2m(input_names=input_names,output_names=output_names,ndays=90)
    builder.load_model_fname = load_model_fname

    fpattern = f"{input_prefix}_*.csv"
    df = read_data(fpattern,input_mask_regex)

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
    for col in builder.output_names:
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
    xvalid_fit_multi(df_in,df_out,builder,
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

model_builders = { "GRUBuilder2" : GRUBuilder2
                 }

def model_builder_from_config(builder_config):
    mbfactory = model_builders[builder_config["builder_name"]]
    builder = mbfactory(**builder_config["args"])
    return builder


def process_config(configfile,proc_steps): 
    """ Configure the model builder subclass and run through training stages 
        Right now the model builder is created at outer scope and does not 
        readjust the model between steps. 
        That may limit some transfer learning options.
    """

    config = read_config(configfile)
    # The model builder config must have a builder_name field    
    # that represents the builder class. The other data 
    # should match the needs of the constructor for that class
    builder_config=config['model_builder_config']
    builder = model_builder_from_config(builder_config)

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
            fit_from_config(builder,**step)

def main():
    #configfile = "restart_config.yml"
    #process_config(configfile,["dsm2_base","dsm2_restart"])

    configfile = "transfer_config.yml"
    #process_config(configfile,["dsm2_base","dsm2.schism"])
    process_config(configfile,["dsm2.schism","base.suisun"])
    process_config(configfile,["base.suisun"])

if __name__ == "__main__":
    main()
    #print(type(GRUBuilder2m).__name__)    

