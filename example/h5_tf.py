import tensorflow as tf
from tensorflow.keras.models import load_model
from casanntra.model_builder import *
from casanntra.staged_learning import read_config,model_builder_from_config
from casanntra.read_data import read_data
from casanntra.scaling import *
configfile = "transfer_config.yml"
config = read_config(configfile)
builder_config = config["model_builder_config"]
builder = model_builder_from_config(builder_config)


output_scales = list(builder.output_names.values())
#  Centralized custom object registration
custom_objects = {"UnscaleLayer": UnscaleLayer,
                               "StackLayer": StackLayer,
                               "ModifiedExponentialDecayLayer": ModifiedExponentialDecayLayer,
                               "masked_mae": masked_mae,
                               "masked_mse": masked_mse,
                               "scaled_masked_mae": ScaledMaskedMAE(output_scales),
                               "scaled_masked_mse": ScaledMaskedMSE(output_scales)}

# Load the .h5 model
fstub = "schism_base.suisun_gru2.h5"
wwstub = fstub.replace(".h5", ".weights.h5")
model = tf.keras.models.load_model(fstub,custom_objects=custom_objects)
model.load_weights(wwstub)
print(model.summary())

# Save the model as a SavedModel
#tf.saved_model.save(model, 'dsm2_base_debug')

model.save("base.suisun_debug", save_format='tf')

