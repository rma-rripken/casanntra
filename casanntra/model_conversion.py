import os
import pandas as pd
import yaml
from tensorflow.keras.models import load_model
from casanntra.model_builder import ModelBuilder, UnscaleLayer
from casanntra.staged_learning import read_config
from casanntra.read_data import read_data


def find_training_step(config, model_name):
    """Finds the training step in the config file that produced the given model."""
    for step in config["steps"]:
        if step["save_model_fname"] == model_name:
            return step
    raise ValueError(f"Error: No training step found for model '{model_name}' in the YAML config.")


def convert_h5_to_tf(config_file, model_h5_path, save_as_tf=True):
    """Converts an .h5 model to TensorFlow's SavedModel format with an UnscaleLayer."""
    config = read_config(config_file)
    model_name = os.path.splitext(os.path.basename(model_h5_path))[0]

    # ✅ Find the corresponding training step
    step = find_training_step(config, model_name)
    
    # ✅ Load the correct ModelBuilder
    builder_config = config["model_builder_config"]
    builder_class = builder_config["builder_name"]
    builder_args = builder_config["args"]

    builder = ModelBuilder(**builder_args)
    
    # ✅ Register required custom objects for model loading
    custom_objects = builder.custom_objects
    
    # ✅ Load the original trained model
    model = load_model(model_h5_path, custom_objects=custom_objects)
    
    # ✅ Wrap model with UnscaleLayer
    wrapped_model = builder.wrap_with_unscale_layer(model)

    # ✅ Save the wrapped model in TensorFlow format
    if save_as_tf:
        tf_model_path = model_name + "_tf"
        wrapped_model.save(tf_model_path, save_format="tf")
        print(f"✅ Model converted and saved at: {tf_model_path}")
        return tf_model_path
    else:
        return wrapped_model


def process_and_predict(model, input_file, output_csv):
    """Processes input data with lagging, prepares for model inference, and saves results."""
    # ✅ Read raw input data
    df_raw = read_data(input_file)

    # ✅ Apply preprocessing (use ModelBuilder's method)
    df_in = builder.calc_antecedent_preserve_cases(df_raw)
    
    # ✅ Restructure for ANN
    df_in = builder.df_by_feature_and_time(df_in).drop(["datetime", "case", "fold"], level="var", axis=1)
    df_in = {name: df_in.loc[:, (name, slice(None))].droplevel("var", axis=1) for name in builder.input_names}

    # ✅ Make predictions
    predictions = model.predict(df_in)
    
    # ✅ Convert predictions to DataFrame
    df_pred = pd.DataFrame(predictions, columns=builder.output_list())
    df_pred.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved at: {output_csv}")


def convert_validate_model(config_file, model_h5_path, input_file):
    """Runs prediction for both the H5 model and TF model, producing output validation CSVs."""
    # ✅ Convert the model to TF format
    tf_model_path = convert_h5_to_tf(config_file, model_h5_path)

    # ✅ Load both models
    h5_model = load_model(model_h5_path, custom_objects=builder.custom_objects)
    tf_model = load_model(tf_model_path, custom_objects={"UnscaleLayer": UnscaleLayer})

    # ✅ Predict with both and save outputs
    h5_output_csv = model_h5_path.replace(".h5", "_h5inputcheck.csv")
    tf_output_csv = tf_model_path + "_tfinputcheck.csv"

    process_and_predict(h5_model, input_file, h5_output_csv)
    process_and_predict(tf_model, input_file, tf_output_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a trained H5 model to TF format and validate with predictions.")
    parser.add_argument("config", type=str, help="Path to the YAML training config file.")
    parser.add_argument("model_h5", type=str, help="Path to the trained H5 model.")
    parser.add_argument("input_file", type=str, help="Path to an input file for model validation.")

    args = parser.parse_args()

    # ✅ Run conversion and validation
    convert_validate_model(args.config, args.model_h5, args.input_file)
