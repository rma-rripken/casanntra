# Reduce regularization
# Larger model
# MSE or weighted MAE

from casanntra.read_data import read_data, compute_scenario_differences
from casanntra.model_builder import *
from casanntra.multi_stage_model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi, bulk_fit
from casanntra.tide_transforms import *
from casanntra.scaling import (ModifiedExponentialDecayLayer,
                              TunableModifiedExponentialDecayLayer)
from keras.models import load_model
from sklearn.decomposition import PCA
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.layers import Layer
import yaml

from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi, bulk_fit
from casanntra.read_data import read_data


model_builders = { "GRUBuilder2" : GRUBuilder2, 
                  "MultiStageModelBuilder": MultiStageModelBuilder }


def fit_from_config(
    builder,
    name,
    input_prefix,       # prefix for input csv files (minus "_1.csv")
    output_prefix, 
    input_mask_regex,   # None     # regex to filter out (not implemented)
    save_model_fname,   # Name for saved model, e.g. dsm2_base_gru2.h5
    load_model_fname,   # Name for loading model or None if it is an original fit
    pool_size,          # Pool size 1 or 2x pool size would cover all the folds
    target_fold_length, # 180d
    pool_aggregation,   # Used to collapse cross-validation folds for large cases
    init_train_rate,    # Initial training rate
    init_epochs,        # Number of epochs for initial warm-up training
    main_train_rate,    # Training rate for main phase
    main_epochs,        # Example: 110
):  
    """Performs a configured sequence of named fitting steps from yaml."""
    
    builder.load_model_fname = load_model_fname

    fpattern = f"{input_prefix}_*.csv"
    df = read_data(fpattern, input_mask_regex)

    # Split the dataset into cross-validation folds
    df_in, df_out = builder.xvalid_time_folds(df, target_fold_length, split_in_out=True)

    # Pool aggregation logic
    if pool_aggregation:
        df_in['fold'] = df_in['fold'] % pool_size
        df_out['fold'] = df_out['fold'] % pool_size


    ref_out_csv = f"{output_prefix}_xvalid_ref_out_unscaled.csv"
    print(ref_out_csv)
    df_out.to_csv(ref_out_csv,float_format="%.3f",
                  date_format="%Y-%m-%dT%H:%M",header=True,index=True)
    
    for col in builder.output_names:
        scale_factor = builder.output_names[col]  # ✅ Get scale factor from YAML
        df_out[col] /= scale_factor  # ✅ Apply scaling before training    
    
    ref_out_csv2 = f"{output_prefix}_xvalid_ref_out_scaled.csv"
    print(ref_out_csv2)
    df_out.to_csv(ref_out_csv,float_format="%.3f",
                  date_format="%Y-%m-%dT%H:%M",header=True,index=True)

    # Perform cross-validation fitting (safe within multithreading)
    xvalid_fit_multi(df_in, df_out, builder,
                     out_prefix=output_prefix, 
                     init_train_rate=init_train_rate,
                     init_epochs=init_epochs, 
                     main_train_rate=main_train_rate, 
                     main_epochs=main_epochs, 
                     pool_size=pool_size)
    
    # Perform bulk fit (final consolidated fit for saving)
    ann = bulk_fit(builder, df_in, df_out,                      
                   output_prefix,                              
                   fit_in=df_in, fit_out=df_out,
                   test_in=df_in, test_out=df_out,
                   init_train_rate=init_train_rate,
                   init_epochs=init_epochs, 
                   main_train_rate=main_train_rate, 
                   main_epochs=main_epochs)
    
    # Save the model
    print(f"Saving model {name} to {save_model_fname}")
    ann.save_weights(save_model_fname.replace(".h5", ".weights.h5"))
    ann.save(save_model_fname, overwrite=True)

def read_config(configfile):
    """Reads YAML config file."""
    with open(configfile) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# ✅ Ensure any builder can be dynamically selected
def model_builder_from_config(builder_config):
    """Factory function to create a model builder instance with required arguments."""
    mbfactory = model_builders[builder_config["builder_name"]]
    
    # ✅ Pass `args` directly, as in the old working version
    builder = mbfactory(**builder_config["args"])
    
    return builder


# ✅ Main function to process YAML steps
def process_config(configfile, proc_steps): 
    """Configure the model builder subclass and run through training stages."""

    config = read_config(configfile)
    builder_config = config['model_builder_config']
    builder = model_builder_from_config(builder_config)

    proc_all = (proc_steps == "all")
    if proc_steps is None:
        raise ValueError("Processing steps or the string 'all' required")
    elif not isinstance(proc_steps, list):
        proc_steps = [proc_steps] if isinstance(proc_steps, str) else proc_steps

    for step in config['steps']:
        if step['name'] in proc_steps or proc_all:

            print("\n\n###############  STEP", step['name'], "############\n")
            for key in step:
                if step[key] == "None":
                    step[key] = None

            # ✅ Remove `builder_args` before passing to fit_from_config()
            step_filtered = {k: v for k, v in step.items() if k != "builder_args"}
            
            # ✅ Just pass `builder_args` without interpreting it
            fit_from_config(builder, **step_filtered)



def verify_data_availability(source_data_prefix, target_data_prefix):
    """
    Checks whether the required datasets exist before proceeding with transfer learning.
    Ensures that both datasets are present when computing scenario differences.
    """

    # ✅ Check if `target_data_prefix` is defined
    if target_data_prefix is None:
        raise ValueError("Error: `target_data_prefix` is required but received None.")

    # ✅ Check if target dataset exists
    target_pattern = f"{target_data_prefix}_*.csv"
    target_files = glob.glob(target_pattern)
    
    if not target_files:
        raise FileNotFoundError(f"Error: No files found for target_data_prefix: {target_data_prefix}")

    # ✅ If `source_data_prefix` is used (e.g., contrastive learning), check it too
    if source_data_prefix is not None:
        source_pattern = f"{source_data_prefix}_*.csv"
        source_files = glob.glob(source_pattern)
        
        if not source_files:
            raise FileNotFoundError(f"Error: No files found for source_data_prefix: {source_data_prefix}")

    print(f"✅ Data verified: Found data for {target_data_prefix}" + 
          (f" and {source_data_prefix}" if source_data_prefix else ""))



