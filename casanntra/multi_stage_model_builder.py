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

        # ✅ Ensure that all components contribute
        total_loss = source_error + 0.0001*target_error + 0.0001*contrast_penalty

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

import tensorflow as tf

def masked_mae(y_true, y_pred):
    """Computes MAE while ignoring NaN values."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    return tf.reduce_mean(tf.abs(tf.boolean_mask(y_true - y_pred, mask)))

def masked_mse(y_true, y_pred):
    """Computes MSE while ignoring NaN values."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true)) & tf.math.logical_not(tf.math.is_nan(y_pred))
    return tf.reduce_mean(tf.square(tf.boolean_mask(y_true - y_pred, mask)))



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
        """Builds or extends an existing model, ensuring UnscaleLayer and loss functions are registered."""
        base_model = self.load_existing_model()  # ✅ Uses centralized model loading

        if base_model:
            input_layer = base_model.input
            feature_extractor = base_model.get_layer('gru_2').output
        else:
            print(f"Creating from scratch")
            prepro_layers = self.prepro_layers(input_layers, input_data)
            x = layers.Lambda(lambda x: tf.stack(x, axis=-1))(prepro_layers)
            x = layers.GRU(units=32, return_sequences=True, activation='sigmoid', name='gru_1')(x)
            feature_extractor = layers.GRU(units=16, return_sequences=False, activation='sigmoid', name='gru_2')(x)
            input_layer = input_layers

        # ✅ Handle Contrastive Learning Model
        if self.transfer_type == "contrastive":
            contrastive_model = self._build_contrastive_model(input_layer, feature_extractor)
            if add_unscaled_output:
                unscaled_outputs = [self._create_unscaled_layer(out) for out in contrastive_model.outputs]
                return Model(inputs=input_layer, outputs=contrastive_model.outputs + unscaled_outputs)
            return contrastive_model

        # ✅ Handle Multi-Task Learning Model (Difference Mode)
        elif self.transfer_type == "difference":
            print("🔍 DEBUG: Calling _build_mtl_model()")
            mtl_model = self._build_mtl_model(input_layer, feature_extractor)
            if add_unscaled_output:
                unscaled_outputs = [self._create_unscaled_layer(out) for out in mtl_model.outputs]
                return Model(inputs=input_layer, outputs=mtl_model.outputs + unscaled_outputs)
            return mtl_model

        # ✅ Default to Standard Model (Direct Transfer)
        else: 
            print("🔍 DEBUG: Calling _build_base_model()")
            base_model = self._build_base_model(input_layer, feature_extractor)
            if add_unscaled_output:
                unscaled_output = self._create_unscaled_layer(base_model.output)
                return Model(inputs=input_layer, outputs=[base_model.output, unscaled_output])
            return base_model

    def _build_base_model(self, input_layer, feature_extractor):
        """Creates a model for original training with only absolute salinity prediction."""
        ec_absolute = layers.Dense(
            units=len(self.output_names), name="out_absolute", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)
        return Model(inputs=input_layer, outputs=ec_absolute)

    def _build_mtl_model(self, input_layer, feature_extractor):
        """Creates the MTL architecture with absolute and scenario difference outputs."""
        ec_absolute = layers.Dense(
            units=len(self.output_names), name="out_absolute", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        ec_diff = layers.Dense(
            units=len(self.output_names), name="out_diff", activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        return Model(inputs=input_layer, outputs=[ec_absolute, ec_diff])

    def _build_contrastive_model(self, input_layer, feature_extractor):
        """Creates a contrastive learning model with scenario separation."""
        ec_base = layers.Dense(
            units=len(self.output_names), name="out_target", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        ec_suisun = layers.Dense(
            units=len(self.output_names), name="out_source", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)


        return Model(inputs=input_layer, outputs=[ec_base, ec_suisun])

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

        # ✅ Step 4: Initialize merged_df with the first DataFrame to ensure all columns exist
        merged_df = aligned_dfs[0].copy()

        # ✅ Step 5: Iteratively update only input columns from all other DataFrames
        for df in aligned_dfs[1:]:
            for col in input_columns:
                if col in df.columns:  # Ensure column exists before updating
                    merged_df[col] = merged_df[col].combine_first(df[col])

        # ✅ Step 6: Create final aligned DataFrames, ensuring original output values are preserved
        final_dfs = []
        for original_df in aligned_dfs:
            df_final = merged_df.copy()

            # 🚨 Preserve only the original output values for this DataFrame
            for col in output_columns:
                if col in original_df.columns:
                    df_final[col] = original_df[col]  # Restore the original output values

            final_dfs.append(df_final)

        return final_dfs

    def fit_model(self, ann, fit_input, fit_output, test_input, test_output, init_train_rate, init_epochs, main_train_rate, main_epochs):
        """Custom fit_model that supports staged learning and multi-output cases with dynamic loss application."""

        # ✅ Determine the correct loss function and output names
        if self.transfer_type == "contrastive":
            loss_function = None  # ✅ Standard MAE loss for main outputs is handled in custom
            output_names = ["out_target", "out_source"]

            # ✅ Create a training-specific wrapper model
            y_true_target = tf.keras.Input(shape=(len(self.output_names),), name="y_true_target")
            y_true_source = tf.keras.Input(shape=(len(self.output_names),), name="y_true_source")

            y_pred_target, y_pred_source = ann.output[:2]  # Get predictions

            # ✅ Attach contrastive loss layer
            y_pred_target, y_pred_source = ContrastiveLossLayer(name="contrastive_loss")([y_true_target, y_true_source, y_pred_target, y_pred_source])

            # ✅ Define training model (only used for training)
            train_model = tf.keras.Model(
                inputs=[ann.input, y_true_target, y_true_source],
                outputs=[y_pred_target, y_pred_source]
            )
            loss_function={"contrastive_loss": lambda y_true, y_pred: tf.reduce_mean(y_pred) * 0},  # ✅ Use correct output name
            metrics = {'contrastive_loss': [masked_mae, masked_mse]}, 
            # 🚨 Explicitly call loss_layer to force it into the computational graph
         
        elif self.transfer_type == "difference":
            loss_function = hybrid_difference_loss  # ✅ Difference-based training
            output_names = ["out_absolute", "out_diff"]
            train_model = ann  # No special wrapper needed
        else:
            print("direct or base training")
            loss_function = "mae"
            metrics = ["mae","mse"]
            output_names = ["output"] #[list(self.output_names.keys())[0]]  # Single-output model
            train_model = ann  # No special wrapper needed
            # ✅ Compile Model (Normal losses for main outputs, `add_loss()` handles contrast)
            loss_dict = {name: loss_function for name in output_names}

        train_model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=0.5),
            loss=loss_function,
            metrics=metrics, 
            run_eagerly=False
        )


        # ✅ Format Inputs for Training
        if self.requires_secondary_data():
            fit_x = [fit_input, fit_output[0], fit_output[1]]
            test_x = [test_input, test_output[0], test_output[1]]
        else:
            fit_x, test_x = fit_input, test_input
        
       
        print("=== DEBUG: Checking Model Summary ===")
        train_model.summary()

        print("=== DEBUG: Initial Training Phase ===")
        # ✅ Initial Training Phase
        history = train_model.fit(
            fit_x, fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_x, test_output),
            verbose=2,
            shuffle=True
        )

        print("=== DEBUG: Main Training Phase ===")
        # ✅ Main Training Phase (Slower Learning Rate)
        if main_epochs and main_epochs > 0:
            train_model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate),
                loss=loss_function,
                metrics=metrics,
                run_eagerly=False
            )
            history = train_model.fit(
                fit_x, fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_x, test_output),
                verbose=2,
                shuffle=True
            )
        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann  # ✅ Base model is returned for inference
