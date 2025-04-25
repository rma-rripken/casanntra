import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

from casanntra.model_builder import UnscaleLayer, StackLayer, ModifiedExponentialDecayLayer
from casanntra.staged_learning import read_config, model_builder_from_config
from casanntra.model_builder import masked_mae, masked_mse, ScaledMaskedMAE, ScaledMaskedMSE

def convert_model(h5_model_path, config_path):
    # ✅ Parse config and initialize builder
    config_data = read_config(config_path)
    output_dir = config_data.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    builder = model_builder_from_config(config_data["model_builder_config"])
    output_scales = list(builder.output_names.values())

    # ✅ Register custom objects
    custom_objects = {
        "UnscaleLayer": UnscaleLayer,
        "StackLayer": StackLayer,
        "ModifiedExponentialDecayLayer": ModifiedExponentialDecayLayer,
        "masked_mae": masked_mae,
        "masked_mse": masked_mse,
        "scaled_masked_mae": ScaledMaskedMAE(output_scales),
        "scaled_masked_mse": ScaledMaskedMSE(output_scales),
    }

    # ✅ Load the .h5 model
    model = load_model(h5_model_path, custom_objects=custom_objects)

    # ✅ Optionally load weights
    weights_path = h5_model_path.replace(".h5", ".weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"✅ Weights loaded from: {weights_path}")
    else:
        print(f"⚠ No separate weights file found at {weights_path} (proceeding with compiled model).")

    print("🔍 Original model loaded:")
    model.summary()

    # ✅ Wrap the model in UnscaleLayer
    #model = builder.wrap_with_unscale_layer(model)

    # ✅ Compile model to ensure output names are preserved
    #model.compile()

    # ✅ Save to TensorFlow SavedModel format
    base_name = os.path.splitext(os.path.basename(h5_model_path))[0]
    save_path = os.path.join(output_dir, base_name)
    model.save(save_path, save_format="tf")
    print(f"✅ Converted model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .h5 model to TensorFlow SavedModel format using config-defined output_dir.")
    parser.add_argument("h5_model_path", help="Path to the .h5 model to convert")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML training config file (e.g. transfer_config.yml)")

    args = parser.parse_args()
    convert_model(args.h5_model_path, args.config)