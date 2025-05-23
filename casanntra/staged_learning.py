from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.multi_stage_model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi, bulk_fit
from casanntra.tide_transforms import *
from casanntra.scaling import (
    ModifiedExponentialDecayLayer,
    TunableModifiedExponentialDecayLayer,
)
from keras.models import load_model
from sklearn.decomposition import PCA
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.layers import Layer
import yaml

from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi, bulk_fit
from casanntra.read_data import read_data
from casanntra.single_or_list import single_or_list

model_builders = {
    "GRUBuilder2": GRUBuilder2,
    "MultiStageModelBuilder": MultiStageModelBuilder,
}


def fit_from_config(
    builder,
    name,
    input_prefix,
    output_prefix,
    input_mask_regex,
    save_model_fname,
    load_model_fname,
    pool_size,
    target_fold_length,
    pool_aggregation,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
):
    """
    Fits a machine learning model using staged training and cross-validation.

    This function orchestrates a structured training sequence, handling data loading,
    cross-validation setup, scaling, model fitting, and final model saving. It supports
    training both standard models and transfer-learning scenarios with secondary datasets.

    Parameters
    ----------
    builder : object
        An instance of a model builder (e.g., `GRUBuilder2` or `MultiStageModelBuilder`).
    name : str
        A descriptive name for the model, used in saving logs and outputs.
    input_prefix : str
        Prefix for input CSV files (excluding `_1.csv` suffix).
    output_prefix : str
        Prefix for output files.
    input_mask_regex : list(str) or None
        Regular expressions to filter input features. Example is [r'schism_base_1.*csv']
    save_model_fname : str
        Filename to save the trained model (excluding `.h5` extension).
    load_model_fname : str or None
        Filename for loading an existing model. If `None`, training starts from scratch.
    pool_size : int
        Number of cross-validation folds or pooling groups. For workstations with 20 cores, 12 is good
    target_fold_length : int
        Length of time-based validation folds in days, typically 180d.
    pool_aggregation : bool
        Whether to aggregate folds to reduce computational load for large datasets. Usually true if the
        number of cases is very large compared to the pool size
    init_train_rate : float
        Initial training learning rate.
    init_epochs : int
        Number of epochs for the initial warm-up training phase.
    main_train_rate : float
        Learning rate for the main training phase.
    main_epochs : int
        Number of epochs for the main training phase.

    Returns
    -------
    None
        The function does not return a value but saves trained models and outputs to disk.

    Examples
    --------
    >>> builder = GRUBuilder2(...)
    >>> fit_from_config(
    ...     builder,
    ...     name="my_model",
    ...     input_prefix="dataset",
    ...     output_prefix="results",
    ...     input_mask_regex=None,
    ...     save_model_fname="my_model",
    ...     load_model_fname=None,
    ...     pool_size=12,
    ...     target_fold_length='180d',
    ...     pool_aggregation=False,
    ...     init_train_rate=0.001,
    ...     init_epochs=10,
    ...     main_train_rate=0.0005,
    ...     main_epochs=100
    ... )



    Notes
    -----
    - Uses `read_data()` to load input data from CSV files.
    - Arguments are often provided via **config read from yaml rather than item by item
    - If `builder.requires_secondary_data()` is `True`, it aligns primary and secondary datasets.
    - Applies `xvalid_time_folds()` to create time-based validation splits.
    - Calls `xvalid_fit_multi()` for cross-validation training.
    - Calls `bulk_fit()` to finalize training and save the model.
    - Saves trained model weights (`.weights.h5`) and full model (`.h5`).

    """

    builder.load_model_fname = load_model_fname

    fpattern = f"{input_prefix}_*.csv"
    df = read_data(fpattern, input_mask_regex)

    # ✅ Handle secondary dataset if required
    if builder.requires_secondary_data():
        source_data_prefix = builder.builder_args.get("source_data_prefix", None)
        if source_data_prefix is None:
            raise ValueError(
                f"{builder.transfer_type} requires source_data_prefix in builder_args"
            )
        source_mask = builder.builder_args.get("source_input_mask_regex", None)

        source_fpattern = f"{source_data_prefix}_*.csv"
        df_source = read_data(source_fpattern, input_mask_regex=source_mask)
        # df.to_csv("df_echo.csv")
        # df_source.to_csv("dfsrc_echo.csv")
        # This takes df and df_source and puts them on a common index that also has
        # common (case,datetime) identity
        df, df_source = builder.pool_and_align_cases([df, df_source])
        # Debugging stuff
        # bigdf = pd.concat([df,df_source], axis = 1)
        # df.to_csv("bigdf.csv",header=True)
        # df_source.to_csv("bigother.csv",header=True)

        df_source_in, df_source_out = builder.xvalid_time_folds(
            df_source, target_fold_length, split_in_out=True
        )

        df_in, df_out = builder.xvalid_time_folds(
            df, target_fold_length, split_in_out=True
        )
        df_in = df_source_in  # Thus far, ANNs have one set of input even when there are multiple outputs
        df_out = [df_out, df_source_out]
    else:
        df_in, df_out = builder.xvalid_time_folds(
            df, target_fold_length, split_in_out=True
        )

    # This works regardless of whether df_out is a list or not
    write_reference_outputs(output_prefix, df_out, builder, is_scaled=False)

    # Pool aggregation logic
    if pool_aggregation:
        df_in["fold"] = df_in["fold"] % pool_size
        if builder.requires_secondary_data():
            df_out[0]["fold"] = df_out[0]["fold"] % pool_size
            df_out[1]["fold"] = df_out[1]["fold"] % pool_size

    # ✅ Scale outputs. Works for single df or list
    # df_out = scale_output(df_out, builder.output_names)

    # ✅ Write scaled reference outputs, works for single dataframe or list
    # write_reference_outputs(output_prefix, df_out, builder, is_scaled=True)

    # Perform cross-validation fitting (safe within multiprocessing)
    xvalid_fit_multi(
        df_in,
        df_out,
        builder,
        out_prefix=output_prefix,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs,
        pool_size=pool_size,
    )

    # Perform bulk fit (final consolidated fit for saving)
    ann = bulk_fit(
        builder,
        df_in,
        df_out,
        output_prefix,
        fit_in=df_in,
        fit_out=df_out,
        test_in=df_in,
        test_out=df_out,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs,
    )

    # Save the model

    """ Debug
    import json
    config = ann.get_config()
    print(json.dumps(config, indent=2))  # Pretty-print model config


    print("Debug: Checking for layers that support masking:")
    for layer in ann.layers:
        if hasattr(layer, "supports_masking") and layer.supports_masking:
            print(f"Layer {layer.name} ({layer.__class__.__name__}) supports masking.")
    for layer in ann.layers:
        if hasattr(layer, 'compute_mask'):
            print(f"Layer {layer.name} has a `compute_mask` method.")
    """

    print(f"Saving model {name} to {save_model_fname+'.h5'}")

    ann.save_weights(save_model_fname + ".weights.h5")
    ann.compile(metrics=None, loss=None)
    ann.save(save_model_fname + ".h5", overwrite=True)


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

    builder_args = builder_config.get("builder_args", {})
    mbfactory = model_builders[builder_config["builder_name"]]
    print(builder_config["args"])
    # ✅ Pass `args` directly, as in the old working version
    builder = mbfactory(**builder_config["args"], **builder_args)

    return builder


@single_or_list("df_out")
def scale_output(df_out, output_scales):
    output_list = list(output_scales)
    for col in output_list:
        scale_factor = output_scales[col]
        df_out.loc[:, col] /= scale_factor
    return df_out


# ✅ Main function to process YAML steps
def process_config(configfile, proc_steps):
    """Configure the model builder subclass and run through training stages."""

    config = read_config(configfile)
    builder_config = config["model_builder_config"]
    builder = model_builder_from_config(builder_config)

    proc_all = proc_steps == "all"
    if proc_steps is None:
        raise ValueError("Processing steps or the string 'all' required")
    elif not isinstance(proc_steps, list):
        proc_steps = [proc_steps] if isinstance(proc_steps, str) else proc_steps

    for step in config["steps"]:
        if step["name"] in proc_steps or proc_all:

            print("\n\n\n\n###############  STEP", step["name"], "############\n")
            # ✅ Extract `builder_args` from step
            builder_args = step.get("builder_args", {})
            print(f"🔍 DEBUG: Builder Args for Step {step['name']} = {builder_args}")

            # ✅ Set new builder_args dynamically instead of recreating the builder
            builder.set_builder_args(builder_args)

            for key in step:
                if step[key] == "None":
                    step[key] = None

            # 🔁 Replace {output_dir} placeholders
            for key in ["save_model_fname", "load_model_fname", "output_prefix"]:
                if key in step and isinstance(step[key], str):
                    step[key] = step[key].replace("{output_dir}", config["output_dir"])

            # ✅ Remove `builder_args` before passing to fit_from_config()
            step_filtered = {k: v for k, v in step.items() if k != "builder_args"}

            # ✅ Just pass `builder_args` without interpreting it

            fit_from_config(builder, **step_filtered)
            # except Exception as e:
            #    print(e)
            #    print("Exception reported by step")
            #    raise


def write_reference_outputs(output_prefix, df_out, builder, is_scaled=False):
    """Writes reference output files for debugging and validation, handling both single and multi-output cases."""

    suffix = "scaled" if is_scaled else "unscaled"

    if isinstance(df_out, list):
        primary_df = df_out[0]
        secondary_df = df_out[1] if builder.requires_secondary_data() else None
    else:
        primary_df = df_out
        secondary_df = None

    # ✅ Write primary reference output
    ref_out_csv = f"{output_prefix}_xvalid_ref_out_{suffix}.csv"
    primary_df.to_csv(
        ref_out_csv,
        float_format="%.3f",
        date_format="%Y-%m-%dT%H:%M",
        header=True,
        index=True,
    )

    # ✅ If secondary data exists, write its reference outputs
    if secondary_df is not None:
        ref_out_csv_secondary = f"{output_prefix}_xvalid_ref_out_secondary_{suffix}.csv"
        secondary_df.to_csv(
            ref_out_csv_secondary,
            float_format="%.3f",
            date_format="%Y-%m-%dT%H:%M",
            header=True,
            index=True,
        )


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
        raise FileNotFoundError(
            f"Error: No files found for target_data_prefix: {target_data_prefix}"
        )

    # ✅ If `source_data_prefix` is used (e.g., contrastive learning), check it too
    if source_data_prefix is not None:
        source_pattern = f"{source_data_prefix}_*.csv"
        source_files = glob.glob(source_pattern)

        if not source_files:
            raise FileNotFoundError(
                f"Error: No files found for source_data_prefix: {source_data_prefix}"
            )

    print(
        f"✅ Data verified: Found data for {target_data_prefix}"
        + (f" and {source_data_prefix}" if source_data_prefix else "")
    )
