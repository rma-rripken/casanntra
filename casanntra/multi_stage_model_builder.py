from keras.models import load_model
from tensorflow.keras import layers, regularizers, Model
import tensorflow as tf
import pandas as pd
from casanntra.model_builder import GRUBuilder2
from tensorflow.keras.layers import Layer


import tensorflow as tf
from tensorflow.keras.layers import Layer

import tensorflow as tf
from tensorflow.keras.layers import Layer

class ContrastiveLossLayer(Layer):
    def call(self, inputs, training=None):
        y_true_target, y_true_source, y_pred_target, y_pred_source = inputs

        # ✅ Compute masks to ignore NaN values
        mask_source = tf.math.logical_not(tf.math.is_nan(y_true_source)) & tf.math.logical_not(tf.math.is_nan(y_pred_source))
        mask_target = tf.math.logical_not(tf.math.is_nan(y_true_target)) & tf.math.logical_not(tf.math.is_nan(y_pred_target))

        # ✅ Compute MAE only for available (non-NaN) values
        source_error = tf.reduce_mean(tf.abs(tf.boolean_mask(y_true_source - y_pred_source, mask_source)))
        target_error = tf.reduce_mean(tf.abs(tf.boolean_mask(y_true_target - y_pred_target, mask_target)))

        # ✅ Provide a fallback zero-loss if no valid values exist
        source_error = tf.where(tf.math.is_finite(source_error), source_error, 0.0)
        target_error = tf.where(tf.math.is_finite(target_error), target_error, 0.0)

        # ✅ Compute contrastive penalty only if both source and target have valid values
        mask_both = mask_source & mask_target
        contrast_penalty = tf.reduce_mean(tf.abs(
            tf.boolean_mask(y_pred_target - y_pred_source, mask_both) - tf.boolean_mask(y_true_target - y_true_source, mask_both)
        ))

        contrast_penalty = tf.where(tf.reduce_any(mask_both), contrast_penalty, 0.0)
        #if tf.reduce_all(tf.equal(0.0, 0.0)):  # 🔥 This ensures contrastive loss is completely removed
        #total_loss = source_error + 0.001 * target_error
        #else:
        total_loss = source_error + target_error + 0.5*contrast_penalty
        # 🔥 Add Debugging Prints
        #tf.print("Batch Loss:", total_loss, "Source:", source_error, "Target:", target_error, "Penalty:", contrast_penalty)

        self.add_loss(tf.reduce_mean(total_loss))

        return y_pred_target, y_pred_source


def hybrid_difference_loss(y_true_target, y_pred_target, y_pred_source):
    """Computes hybrid difference loss where source model is fixed, and target model must match both its own truth and the difference."""
    
    # ✅ Ensure NaN robustness
    mask_target = tf.math.logical_not(tf.math.is_nan(y_true_target)) & tf.math.logical_not(tf.math.is_nan(y_pred_target))
    
    # ✅ Target accuracy penalty
    target_error = tf.reduce_mean(tf.abs(tf.boolean_mask(y_true_target, mask_target) - tf.boolean_mask(y_pred_target, mask_target)))
    
    # ✅ Difference consistency penalty (forces ANN to predict correct differences)
    difference_penalty = tf.reduce_mean(tf.abs((y_pred_target - y_pred_source) - (y_true_target - y_pred_source)))

    # ✅ Weighted sum of errors
    return target_error + 0.5 * difference_penalty

def masked_mae(y_true, y_pred):
    """Computes MAE while ignoring NaN values and ensuring NaN-safe computation."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    valid_values = tf.boolean_mask(y_true - y_pred, mask)

    # ✅ If there are no valid values, return 0.0 instead of NaN
    return tf.cond(
        tf.size(valid_values) > 0,
        lambda: tf.reduce_mean(tf.abs(valid_values)),
        lambda: tf.constant(0.0, dtype=tf.float32)  # ✅ Return 0 loss if no valid values
    )

def masked_mse(y_true, y_pred):
    """Computes MAE while ignoring NaN values and ensuring NaN-safe computation."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    valid_values = tf.boolean_mask(y_true - y_pred, mask)

    # ✅ If there are no valid values, return 0.0 instead of NaN
    return tf.cond(
        tf.size(valid_values) > 0,
        lambda: tf.reduce_mean(tf.square(valid_values)),
        lambda: tf.constant(1e-7, dtype=tf.float32)  # ✅ Avoid zero loss, ensure gradients exist
    )

def masked_mae_target(y_true, y_pred):
    """Computes MAE for the target output while ignoring NaN values."""
    y_true_target = y_true[0]  # Extract target component
    y_pred_target = y_pred[0]  # Extract target prediction

    mask = tf.math.logical_not(tf.math.is_nan(y_true_target)) & tf.math.logical_not(tf.math.is_nan(y_pred_target))
    return tf.reduce_mean(tf.abs(tf.boolean_mask(y_true_target - y_pred_target, mask)))

def masked_mae_source(y_true, y_pred):
    """Computes MAE for the source output while ignoring NaN values."""
    y_true_source = y_true[1]  # Extract source component
    y_pred_source = y_pred[1]  # Extract source prediction

    mask = tf.math.logical_not(tf.math.is_nan(y_true_source)) & tf.math.logical_not(tf.math.is_nan(y_pred_source))
    return tf.reduce_mean(tf.abs(tf.boolean_mask(y_true_source - y_pred_source, mask)))

def contrastive_penalty(y_true, y_pred):
    """Computes contrastive loss separately for visualization."""
    y_true_target, y_true_source = y_true  # Extract true values
    y_pred_target, y_pred_source = y_pred  # Extract predictions

    mask_source = tf.math.logical_not(tf.math.is_nan(y_true_source)) & tf.math.logical_not(tf.math.is_nan(y_pred_source))
    mask_target = tf.math.logical_not(tf.math.is_nan(y_true_target)) & tf.math.logical_not(tf.math.is_nan(y_pred_target))

    # Compute contrastive penalty only if both target and source have valid values
    mask_both = mask_source & mask_target
    contrast_penalty = tf.reduce_mean(tf.abs(
        tf.boolean_mask(y_pred_target - y_pred_source, mask_both) - tf.boolean_mask(y_true_target - y_true_source, mask_both)
    ))
    return tf.where(tf.reduce_any(mask_both), contrast_penalty, 0.0)



# multi_stage_model_builder.py
class MultiStageModelBuilder(GRUBuilder2):

    def __init__(self, input_names, output_names, ndays=90):
        """Multi-stage model builder that supports flexible transfer learning options."""
        super().__init__(input_names, output_names, ndays)

        # ✅ Register additional loss functions required for staged training
        self.register_custom_object("ContrastiveLossLayer", ContrastiveLossLayer)
        self.register_custom_object("hydbrid_difference_loss", hybrid_difference_loss)

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
            x = layers.Lambda(lambda x: tf.stack(x, axis=-1))(prepro_layers)
            x = layers.GRU(units=32, return_sequences=True, activation='sigmoid', name='gru_1')(x)
            feature_extractor = layers.GRU(units=16, return_sequences=False, activation='sigmoid', name='gru_2')(x)
            input_layer = input_layers
            self.old_dense_layer = None
            self.old_weights = None

        # ✅ Contrastive Learning Model
        if self.transfer_type == "contrastive":
            if self.transfer_type == "contrastive":
                out_target_layer = layers.Dense(units=len(self.output_names), name="out_target", activation='elu')
                out_source_layer = layers.Dense(units=len(self.output_names), name="out_source", activation='elu')
                out_source = out_source_layer(feature_extractor)
                out_target = out_target_layer(feature_extractor)
                out_source_layer.set_weights(self.old_weights)
                out_target_layer.set_weights(self.old_weights)
                out_target_layer.trainable = True   # todo 
            model = Model(inputs=input_layer, outputs=[out_target, out_source])       
            
            #model.add_loss(masked_mae_target)
            #model.add_loss(masked_mae_source)
            #model.add_loss(contrastive_penalty)

            # ✅ Attach metrics (computed at runtime)
            #model.add_metric(masked_mae_target, name="mae_target")
            #model.add_metric(masked_mae_source, name="mae_source")
            #model.add_metric(contrastive_penalty, name="contrastive_loss")

            return model

        # ✅ Multi-Task Learning (Difference)
        elif self.transfer_type == "difference":
            out_absolute = layers.Dense(units=len(self.output_names), name="out_absolute", activation='elu')(feature_extractor)
            out_diff = layers.Dense(units=len(self.output_names), name="out_diff", activation='linear')(feature_extractor)
            
            model = Model(inputs=input_layer, outputs=[out_absolute, out_diff])

            model.add_loss(masked_mae(self.output_names[0], out_absolute))
            model.add_loss(masked_mae(self.output_names[1], out_diff))
            model.compile(optimizer='adam')
            model.add_metric(masked_mse(self.output_names[0], out_absolute), name="mse_absolute")
            model.add_metric(masked_mse(self.output_names[1], out_diff), name="mse_diff")

            return model

        # ✅ Default Direct Mode
        else:
            out_absolute = layers.Dense(units=len(self.output_names), name="out_absolute", activation='elu')(feature_extractor)

            model = Model(inputs=input_layer, outputs=out_absolute)

            model.add_loss(masked_mae(self.output_names[0], out_absolute))
            model.add_metric(masked_mse(self.output_names[0], out_absolute), name="mse_absolute")

            return model

    def get_loss_function(self):
        """Returns the correct loss function based on transfer_type."""
        if self.transfer_type == "mtl":
            return weighted_mtl_loss
        if self.transfer_type == "contrastive":
            return hybrid_contrastive_loss_keras  # ✅ Both models train together
        if self.transfer_type == "difference":
            return hybrid_difference_loss   # ✅ Only the target model trains
        return "mae"

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
        """Handles model training in two stages: initial training and fine-tuning."""

        train_model = ann  # Model with loss functions already attached
        # todo: make this configurable
        for layer in ann.layers:
            if layer.name in ["gru_1", "gru_2"]:
                layer.trainable = False

        train_model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            run_eagerly=False,  # No need for eager execution
            loss =  {"out_target" : masked_mae, "out_source": masked_mae}, #{"out_source": masked_mae},              # todo 
            metrics = {"out_target" : [masked_mae,masked_mse], "out_source": [masked_mae,masked_mse]}
        )

        # ✅ Initial Training Phase (larger learning rate)
        print("=== DEBUG: Initial Training Phase ===")
        history = train_model.fit(
            fit_input, fit_output,  # ⬅️ Only the actual feature input is used
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
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
            print("✅ Feature layers (gru_1, gru_2) are UNFROZEN for main training.")            
            print("=== DEBUG: Main Training Phase ===")
            train_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate,
                                                                     clipnorm=0.5))
            train_model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate, clipnorm=0.5),
                run_eagerly=False,  # No need for eager execution
                loss =  {"out_target" : masked_mae, "out_source": masked_mae}, #{"out_source": masked_mae},              # todo 
                metrics = {"out_target" : [masked_mae,masked_mse], "out_source": [masked_mae,masked_mse]}
            )
            history = train_model.fit(
                fit_input, fit_output,  # ⬅️ Again, only the real input data
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True
            )

        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann
