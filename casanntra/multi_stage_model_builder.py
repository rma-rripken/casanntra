from keras.models import load_model
from tensorflow.keras import layers, regularizers, Model
import tensorflow as tf
import pandas as pd
from casanntra.model_builder import GRUBuilder2,StackLayer
import numpy  as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


##@tf.keras.utils.register_keras_serializable
def masked_mae(y_true, y_pred):
    # Mask NaN values, replace by 0
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
   
    # Calculate absolute differences
    absolute_differences = tf.abs(y_true - y_pred)
    
    # Compute the mean, ignoring potential NaN values (if any remain after replacement)
    mae = tf.reduce_mean(absolute_differences)
    
    return mae

#@tf.keras.utils.register_keras_serializable
def masked_mse(y_true, y_pred):
    # Mask NaN values, replace by 0
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
   
    # Calculate absolute differences
    absolute_differences = tf.square(y_true - y_pred)
    
    # Compute the mean, ignoring potential NaN values (if any remain after replacement)
    mae = tf.reduce_mean(absolute_differences)
    
    return mae

def masked_mae1(y_true, y_pred):
    """Computes MAE while ignoring NaN values and ensuring NaN-safe computation."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    valid_values = tf.boolean_mask(y_true - y_pred, mask)

    # ✅ If there are no valid values, return 0.0 instead of NaN
    return tf.cond(
        tf.size(valid_values) > 0,
        lambda: tf.reduce_mean(tf.abs(valid_values)),
        lambda: tf.constant(0.0, dtype=tf.float32)  # ✅ Return 0 loss if no valid values
    )

def masked_mse2(y_true, y_pred):
    """Computes MAE while ignoring NaN values and ensuring NaN-safe computation."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    valid_values = tf.boolean_mask(y_true - y_pred, mask)

    # ✅ If there are no valid values, return 0.0 instead of NaN
    return tf.cond(
        tf.size(valid_values) > 0,
        lambda: tf.reduce_mean(tf.square(valid_values)),
        lambda: tf.constant(1e-7, dtype=tf.float32)  # ✅ Avoid zero loss, ensure gradients exist
    )



# multi_stage_model_builder.py
class MultiStageModelBuilder(GRUBuilder2):

    def __init__(self, input_names, output_names, ndays=90):
        """Multi-stage model builder that supports flexible transfer learning options."""
        super().__init__(input_names, output_names, ndays)

        # ✅ Register additional loss functions required for staged training
        self.register_custom_object("masked_mae", masked_mae)
        self.register_custom_object("masked_mse", masked_mse)

    def set_builder_args(self, builder_args):
        """Allows builder_args to be updated dynamically between steps."""
        self.builder_args = builder_args
        ttype = builder_args.get("transfer_type", None)
        if ttype is None or ttype == "None":
            ttype = "direct" # todo None conversion is done centrally for top level keys in yml, should do it at other depths
        self.transfer_type = ttype
        if self.transfer_type is not None:
            transfer_opts = ["direct","difference","contrastive"]
            if self.transfer_type not in transfer_opts:
                raise ValueError(f"Transfer type {self.transfer_type} not in available options: {transfer_opts}")        

    def num_outputs(self):
        """Multi-output model: primary output + secondary ANN output"""
        nout = 2 if self.transfer_type in ["contrastive", "difference"] else 1
        return nout


    def build_model(self, input_layers, input_data, add_unscaled_output=False):
        """Builds the ANN model with explicit loss functions and metric tracking."""
        
        base_model = self.load_existing_model()

        if base_model:
            input_layer = base_model.input
            feature_extractor = base_model.get_layer('gru_2').output
            self.old_dense_layer = base_model.get_layer("out_absolute")  # Ensure correct layer name
            self.old_weights = self.old_dense_layer.get_weights()
        else:
            prepro_layers = self.prepro_layers(input_layers, input_data)
            x = StackLayer(name="stack_layer")(prepro_layers)
            x = layers.GRU(units=32, return_sequences=True, activation='sigmoid', name='gru_1')(x)
            feature_extractor = layers.GRU(units=16, return_sequences=False, activation='sigmoid', name='gru_2')(x)
            input_layer = input_layers
            self.old_dense_layer = None
            self.old_weights = None

        # ✅ Contrastive Learning Model
        if self.transfer_type == "contrastive":


            # ✅ Define explicit `y_true` placeholders for training only
            y_true_target = layers.Input(shape=(len(self.output_names),), name="y_true_target")
            y_true_source = layers.Input(shape=(len(self.output_names),), name="y_true_source")           

            # ✅ Prevent Keras from computing gradients for these inputs
            y_true_target = tf.stop_gradient(y_true_target)
            y_true_source = tf.stop_gradient(y_true_source)



            out_target_layer = layers.Dense(units=len(self.output_names), name="out_target", activation='elu')
            out_source_layer = layers.Dense(units=len(self.output_names), name="out_source", activation='elu')
            out_source = out_source_layer(feature_extractor)
            out_target = out_target_layer(feature_extractor)
            out_source_layer.set_weights(self.old_weights)
            out_target_layer.set_weights(self.old_weights)
            out_target_layer.trainable = True   # todo investigate further

            out_contrast = layers.Subtract(name="out_contrast")([out_target, out_source])
            ann = Model(inputs=input_layer, outputs=[out_target, out_source, out_contrast])    
            return ann


        # ✅ Default Direct Mode
        else:
            out_absolute = layers.Dense(units=len(self.output_names), name="out_absolute", activation='elu')(feature_extractor)
            model = Model(inputs=input_layer, outputs=out_absolute)
            return model


    def requires_secondary_data(self):
        """Returns True if transfer learning requires a second dataset."""
        requires_2nd =  self.transfer_type in ["difference", "contrastive"]
        return requires_2nd


    def pool_and_align_cases(self, dataframes):
        """Aligns and pools multiple DataFrames so that they all contain the union of (case, datetime) combinations.
        
        Ensures:
        1) Input dates per case overlap **or** one DataFrame is empty within each case.
        2) Datetimes within cases are contiguous so that the union of datetimes are contiguous.
        3) Returns reindexed DataFrames, one per incoming df, where (case, datetime) exist in both.
        4) "Input" columns (in list(self.input_names)) must have data. 
        a. If (case, datetime) are shared among DataFrames, the values should match with high precision.
        b. If a row matching (case, datetime) is missing in one dataset, 
            - Output columns (self.output_names) should be filled with NaN.
            - Input columns (self.input_names) should be filled from the first available non-missing value.
        
        Args:
            dataframes (list of pd.DataFrame): List of DataFrames to be pooled and aligned.

        Returns:
            list of pd.DataFrame: Aligned DataFrames with consistent (case, datetime) indexing. 
            'datetime' and 'case' will be columns, and the index will be an integer.
        """

        # ✅ Step 1: Collect all unique (case, datetime) combinations
        all_case_datetime = pd.concat([df[['case', 'datetime']] for df in dataframes]).drop_duplicates().sort_values(['case', 'datetime'])

        # ✅ Step 2: Create aligned versions of each DataFrame
        aligned_dfs = [all_case_datetime.merge(df, on=['case', 'datetime'], how='left') for df in dataframes]


        # ✅ Step 3: Identify input and output columns
        input_columns = list(self.input_names)
        output_columns = list(self.output_names)

        # Step 4: Initialize merged_df with the first DataFrame to ensure all columns exist
        merged_df = aligned_dfs[0].copy()

        # Step 5: Iteratively update only input columns from all other DataFrames
        for df in aligned_dfs[1:]:
            for col in input_columns:
                if col in df.columns:  # Ensure column exists before updating
                    merged_df[col] = merged_df[col].combine_first(df[col])

        # Step 6: Create final aligned DataFrames, ensuring original output values are preserved
        final_dfs = []
        for original_df in aligned_dfs:
            df_final = merged_df.copy()[["datetime","case","model","scene"] + input_columns + output_columns]
            df_final['model'] = original_df.model
            df_final['scene'] = original_df.scene
            #  Preserve only the original output values for this DataFrame
            for col in output_columns:
                
                if col in original_df.columns:
                    df_final[col] = original_df[col]  # Restore the original output values
            final_dfs.append(df_final)
        return final_dfs

    def fit_model(self, ann, fit_input, fit_output, test_input, test_output, init_train_rate, init_epochs, main_train_rate, main_epochs):
        if self.transfer_type == "contrastive":
            return self._fit_model_contrastive(ann, fit_input, fit_output, 
                                                    test_input, test_output, 
                                                    init_train_rate, init_epochs, 
                                                    main_train_rate, main_epochs)
        else:
            return self._fit_model_direct(ann, fit_input, fit_output, 
                                          test_input, test_output, 
                                          init_train_rate, init_epochs, 
                                          main_train_rate, main_epochs)            

    def _fit_model_contrastive(self, ann, fit_input, fit_output, test_input, test_output, init_train_rate, init_epochs, main_train_rate, main_epochs):
        """Handles model training in two stages: initial training and fine-tuning."""
        
        
        contrastive_target = fit_output[0] - fit_output[1]  # Precomputed contrast
        contrast_weight = 0.50

        # ✅ Mask NaNs properly
        contrastive_target[np.isnan(fit_output[0]) | np.isnan(fit_output[1])] = np.nan

        train_y = {
            "out_target": fit_output[0],
            "out_source": fit_output[1],
            "out_contrast": contrastive_target  # ✅ Treated just like a regular target
        }

        contrastive_test = test_output[0] - test_output[1]  # Precomputed contrast
        # ✅ Validation labels
        test_y = {
            "out_target": test_output[0],
            "out_source": test_output[1],
            "out_contrast": test_output[0] - test_output[1]
        }
        test_y["out_contrast"][np.isnan(test_output[0]) | np.isnan(test_output[1])] = np.nan  # ✅ Mask NaNs

        # todo: make this configurable
        for layer in ann.layers:
            if layer.name in ["gru_1", "gru_2"]:
                layer.trainable = False

        # todo: this is unique to contrastive
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            run_eagerly=False,  # No need for eager execution
            loss =  {"out_target" : masked_mae, 
                     "out_source":  masked_mae,
                     "out_contrast": masked_mae},
            loss_weights={  # ✅ Assign loss weights to emulate λ effect
                "out_target": 1.0,
                "out_source": 1.0,
                "out_contrast": contrast_weight  # ✅ Contrastive loss weight
            },                     
            metrics = {"out_target" :  [masked_mae,masked_mse], 
                       "out_source":   [masked_mae,masked_mse],
                       "out_contrast": [masked_mae, masked_mse]})

        # ✅ Initial Training Phase (larger learning rate)
        print("=== DEBUG: Initial Training Phase ===")
        history = ann.fit(
            fit_input, train_y,  # ⬅️ Only the actual feature input is used
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_y),   # todo: train_x and test_x are for input-augmented
            verbose=2,
            shuffle=True
        )

        # ✅ Main Training Phase (fine-tuning with a lower learning rate)
        if main_epochs and main_epochs > 0:
            # Unfreeze feature layers before main training. 
            # Todo: make this an options
            for layer in ann.layers:
                if layer.name in ["gru_1", "gru_2"]:
                    layer.trainable = True
            print(" Feature layers (gru_1, gru_2) are UNFROZEN for main training.")            
            print("=== DEBUG: Main Training Phase ===")

            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
                run_eagerly=False,  # No need for eager execution
                loss =  {"out_target" : masked_mae, 
                        "out_source":  masked_mae,
                        "out_contrast": masked_mae},
                loss_weights={  # ✅ Assign loss weights to emulate λ effect
                    "out_target": 1.0,
                    "out_source": 1.0,
                    "out_contrast": contrast_weight  # ✅ Contrastive loss weight
                },                        
                metrics = {"out_target" :  [masked_mae,masked_mse], 
                        "out_source":   [masked_mae,masked_mse],
                        "out_contrast": [masked_mae, masked_mse]})

            history = ann.fit(
                fit_input, train_y,  # ⬅️ Again, only the real input data
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_y),
                verbose=2,
                shuffle=True
            )

        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann
    

    def _fit_model_direct(self, ann, fit_input, fit_output, test_input, test_output, init_train_rate, init_epochs, main_train_rate, main_epochs):

        """Custom fit_model that supports staged learning and multi-output cases with dynamic loss application."""

        print("direct or base training")
        loss_function = "mae"
        metrics = ["mae","mse"]
        output_names = ["output"] #[list(self.output_names.keys())[0]]  # Single-output model
        train_model = ann  # No special wrapper needed
        # ✅ Compile Model (Normal losses for main outputs, `add_loss()` handles contrast)
        loss_dict = {name: loss_function for name in output_names}

        # todo: make this configurable
        #for layer in ann.layers:
        #    if layer.name in ["gru_1", "gru_2"]:
        #        layer.trainable = False

        train_model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            loss=loss_function,
            metrics=metrics, 
            run_eagerly=False
        )
    

        print("=== DEBUG: Initial Training Phase ===")
        # ✅ Initial Training Phase
        history = train_model.fit(
            fit_input, fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
        )

        print("=== DEBUG: Main Training Phase ===")
        # Main Training Phase (Slower Learning Rate)
        if main_epochs and main_epochs > 0:


            train_model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate),
                loss=loss_function,
                metrics=metrics,
                run_eagerly=False
            )
            history = train_model.fit(
                fit_input, fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True
            )
        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann  # ✅ Base model is returned for inference

