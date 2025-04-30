from keras.models import load_model
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.layers import Reshape, Concatenate
import tensorflow as tf
import pandas as pd
from casanntra.model_builder import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer




# multi_stage_model_builder.py
class MultiStageModelBuilder(GRUBuilder2):

    def __init__(self, input_names, output_names, ndays=90):
        """Multi-stage model builder that supports flexible transfer learning options."""
        super().__init__(input_names, output_names, ndays)
        output_scales = list(self.output_names.values())
        # ✅ Register additional loss functions required for staged training



    def set_builder_args(self, builder_args):
        """Allows builder_args to be updated dynamically between steps."""
        self.builder_args = builder_args
        self.transfer_type = builder_args.get("transfer_type", "direct")
        if self.transfer_type == "None": 
            self.transfer_type = None

        # ✅ Only load contrast_weight if contrastive mode is selected
        if self.transfer_type == "contrastive":
            self.contrast_weight = float(builder_args.get("contrast_weight", 1.0))
        else:
            self.contrast_weight = None  #  Prevents accidental use in non-contrastive cases
        print("Transfer type: ",self.transfer_type)
        print(self.transfer_type is None)
        print(self.transfer_type == "None")

        if self.transfer_type is not None:
            transfer_opts = ["direct", "difference", "contrastive"]
            if self.transfer_type not in transfer_opts:
                raise ValueError(
                    f"Transfer type {self.transfer_type} not in available options: {transfer_opts}"
                )



    def num_outputs(self):
        """Multi-output model: primary output + secondary ANN output"""
        nout = 3 if self.transfer_type in ["contrastive", "difference"] else 1
        return nout

    def build_model(self, input_layers, input_data):
        """Builds the ANN model with explicit loss functions and metric tracking."""

        base_model = self.load_existing_model()


        if base_model:            
            # Load the base model and rename its dense output layer
            if isinstance(base_model.input, list):
                input_layer = {layer.name: layer for layer in base_model.input}  # ✅ Dictionary for multi-input models
            else:
                input_layer = base_model.input  # ✅ Single tensor for single-input models
            feature_extractor = base_model.get_layer("gru_2").output
            try:
                self.old_dense_layer = base_model.get_layer("out_scaled")
                self.old_weights = self.old_dense_layer.get_weights()
            except ValueError:
                self.old_dense_layer = None
                self.old_weights = None
                raise
        else:
            self.old_dense_layer = None
            self.old_weights = None
            prepro_layers = self.prepro_layers(input_layers, input_data)
            expanded_inputs = [Reshape((self.ndays, 1))(tensor) for tensor in prepro_layers]

            # Concatenate along the last axis to get shape (batch_size, ntime, nfeature)
            x = Concatenate(axis=-1,name="stacked")(expanded_inputs)

            x = layers.GRU(   # was LSTM 32
                units=32, return_sequences=True, activation="sigmoid", name="gru_1"
            )(x)
            feature_extractor = layers.GRU(  # todo: was LSTM 16
                units=16, return_sequences=False, activation="sigmoid", name="gru_2"
            )(x)
            input_layer = input_layers
            self.old_dense_layer = None
            self.old_weights = None

        # ✅ Contrastive Learning Model
        if self.transfer_type == "contrastive":
            # Explicitly define scaled output layers first (clearly distinct layer names)
            out_target_layer = layers.Dense(
              units=len(self.output_names), activation="elu", name="target_scaled"
                )
            out_source_layer = layers.Dense(
                 units=len(self.output_names), name="source_scaled", activation="elu"
             )

            # Apply these layers explicitly to the feature extractor
            out_target_scaled = out_target_layer(feature_extractor)
            out_source_scaled = out_source_layer(feature_extractor)
            out_source_layer.set_weights(self.old_weights)
            out_target_layer.set_weights(self.old_weights)
            out_target_layer.trainable = True  # todo investigate further


            # Explicitly apply unscaling immediately after reusing layers
            output_scales = list(self.output_names.values())
            out_target_unscaled = UnscaleLayer(output_scales, name="out_target_unscaled")(out_target_scaled)
            out_source_unscaled = UnscaleLayer(output_scales, name="out_source_unscaled")(out_source_scaled)
            out_contrast_unscaled = layers.Subtract(name="out_contrast_unscaled")(
                [out_target_unscaled, out_source_unscaled]
            )

            # Explicit minimal correction in model outputs (CRITICAL FIX)
            ann = Model(
                inputs=input_layer, 
                outputs={
                    "out_target_unscaled": out_target_unscaled,
                    "out_source_unscaled": out_source_unscaled,
                    "out_contrast_unscaled": out_contrast_unscaled
                }
            )
            return ann

        # ✅ Default Direct Mode
        else:
            # Dense outputs in scaled units first
            scaled_output = layers.Dense(len(self.output_names), activation="elu", name="out_scaled")(feature_extractor)

            # explicitly add unscaling directly in base model
            # self.output_names.values() are the scaling factors
            unscaled_output = UnscaleLayer(list(self.output_names.values()), name="out_unscaled")(scaled_output)

            model = Model(inputs=input_layer, 
                          outputs={"out_unscaled": unscaled_output})

            # todo this needs to get moved to model builder or it likely will not work with fitting
            return model

    def requires_secondary_data(self):
        """Returns True if transfer learning requires a second dataset."""
        requires_2nd = self.transfer_type in ["difference", "contrastive"]
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
        all_case_datetime = (
            pd.concat([df[["case", "datetime"]] for df in dataframes])
            .drop_duplicates()
            .sort_values(["case", "datetime"])
        )

        # ✅ Step 2: Create aligned versions of each DataFrame
        aligned_dfs = [
            all_case_datetime.merge(df, on=["case", "datetime"], how="left")
            for df in dataframes
        ]

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
            df_final = merged_df.copy()[
                ["datetime", "case", "model", "scene"] + input_columns + output_columns
            ]
            df_final["model"] = original_df.model
            df_final["scene"] = original_df.scene
            #  Preserve only the original output values for this DataFrame
            for col in output_columns:

                if col in original_df.columns:
                    df_final[col] = original_df[
                        col
                    ]  # Restore the original output values
            final_dfs.append(df_final)
        return final_dfs

    def fit_model(
        self,
        ann,
        fit_input,
        fit_output,
        test_input,
        test_output,
        init_train_rate,
        init_epochs,
        main_train_rate,
        main_epochs,
    ):
        if self.transfer_type == "contrastive":
            return self._fit_model_contrastive(
                ann,
                fit_input,
                fit_output,
                test_input,
                test_output,
                init_train_rate,
                init_epochs,
                main_train_rate,
                main_epochs,
            )
        else:
            return self._fit_model_direct(
                ann,
                fit_input,
                fit_output,
                test_input,
                test_output,
                init_train_rate,
                init_epochs,
                main_train_rate,
                main_epochs,
            )

    def _fit_model_contrastive(
        self,
        ann,
        fit_input,
        fit_output,
        test_input,
        test_output,
        init_train_rate,
        init_epochs,
        main_train_rate,
        main_epochs,
    ):
        """Handles model training in two stages: initial training and fine-tuning."""

        contrastive_target = fit_output[0] - fit_output[1]  # Precomputed contrast
        contrast_weight = self.contrast_weight if self.contrast_weight is not None else 1.0

        # ✅ Mask NaNs properly
        contrastive_target[np.isnan(fit_output[0]) | np.isnan(fit_output[1])] = np.nan

        train_y = {
            "out_target_unscaled": fit_output[0],
            "out_source_unscaled": fit_output[1],
            "out_contrast_unscaled": contrastive_target,  # ✅ Treated just like a regular target
        }

        contrastive_test = test_output[0] - test_output[1]  # Precomputed contrast
        # ✅ Validation labels
        test_y = {
            "out_target_unscaled": test_output[0],
            "out_source_unscaled": test_output[1],
            "out_contrast_unscaled": test_output[0] - test_output[1],
        }
        test_y["out_contrast_unscaled"][
            np.isnan(test_output[0]) | np.isnan(test_output[1])
        ] = np.nan  # ✅ Mask NaNs

        # todo: make this configurable
        for layer in ann.layers:
            if layer.name in ["gru_1", "gru_2"]:
                layer.trainable = False

        output_scales = list(self.output_names.values())

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(
                learning_rate=init_train_rate, clipnorm=0.5
            ),
            run_eagerly=False,  # No change needed here
            loss={
                    "out_target_unscaled":  ScaledMaskedMAE(output_scales),
                    "out_source_unscaled":  ScaledMaskedMAE(output_scales),
                    "out_contrast_unscaled": ScaledMaskedMAE(output_scales),
                    },
            loss_weights={
                    "out_target_unscaled": 1.0,
                    "out_source_unscaled": 1.0,
                    "out_contrast_unscaled": contrast_weight,
                    },
            metrics={
                    "out_target_unscaled":  [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)],
                    "out_source_unscaled":  [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)],
                    "out_contrast_unscaled": [masked_mae, masked_mse],
                     },
        )

        # ✅ Initial Training Phase (larger learning rate)
        print("=== DEBUG: Initial Training Phase ===")

        print("Expected model outputs:", ann.output_names)
        print("Provided train_y keys:", train_y.keys())
        assert set(ann.output_names) == set(train_y.keys()), "Mismatch between model outputs and training labels!"

        history = ann.fit(
            fit_input,
            train_y,  # ⬅️ Only the actual feature input is used
            epochs=init_epochs,
            batch_size=64,
            validation_data=(
                test_input,
                test_y,
            ),  # todo: train_x and test_x are for input-augmented
            verbose=2,
            shuffle=True,
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
            output_scales = list(self.output_names.values())
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(
                learning_rate=main_train_rate, clipnorm=0.5
                ),
                run_eagerly=False,
                loss={
                    "out_target_unscaled":  ScaledMaskedMAE(output_scales),
                    "out_source_unscaled":  ScaledMaskedMAE(output_scales),
                    "out_contrast_unscaled": ScaledMaskedMAE(output_scales),
                },
                loss_weights={
                    "out_target_unscaled": 1.0,
                    "out_source_unscaled": 1.0,
                    "out_contrast_unscaled": contrast_weight,
                },
                metrics={
                    "out_target_unscaled":  [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)],
                    "out_source_unscaled":  [ScaledMaskedMAE(output_scales), ScaledMaskedMSE(output_scales)],
                    "out_contrast_unscaled": [masked_mae, masked_mse],
                },
            )

            history = ann.fit(
                fit_input,
                train_y,  # ⬅️ Again, only the real input data
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_y),
                verbose=2,
                shuffle=True,
            )

        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann

    def _fit_model_direct(
        self,
        ann,
        fit_input,
        fit_output,
        test_input,
        test_output,
        init_train_rate,
        init_epochs,
        main_train_rate,
        main_epochs,
    ):
        """Custom fit_model that supports staged learning and multi-output cases with dynamic loss application."""

        print("direct or base training")
        loss_function = "mae"
        output_names = [
            "output"
        ]  # [list(self.output_names.keys())[0]]  # Single-output model
        train_model = ann  # No special wrapper needed
        # ✅ Compile Model (Normal losses for main outputs, `add_loss()` handles contrast)
        loss_dict = {name: loss_function for name in output_names}

        output_scales = list(self.output_names.values())

        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(
                learning_rate=init_train_rate, clipnorm=0.5
            ),
            loss={"out_unscaled": ScaledMaskedMAE(output_scales)},
            metrics={
                "out_unscaled": [
                    ScaledMaskedMAE(output_scales),
                    ScaledMaskedMSE(output_scales)
                ]
            },
                run_eagerly=False,
            )

        print("=== DEBUG: Initial Training Phase ===")
        # ✅ Initial Training Phase
        history = train_model.fit(
            fit_input,
            fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True,
        )

        print("=== DEBUG: Main Training Phase ===")
        # Main Training Phase (Slower Learning Rate)
        if main_epochs and main_epochs > 0:

            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(
                    learning_rate=main_train_rate, clipnorm=0.5
                ),
                loss={"out_unscaled": ScaledMaskedMAE(output_scales)},
                metrics={
                    "out_unscaled": [
                        ScaledMaskedMAE(output_scales),
                        ScaledMaskedMSE(output_scales)
                    ]
                },
                run_eagerly=False,
            )
            history = train_model.fit(
                fit_input,
                fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True,
            )
        print("=== DEBUG: Completed Main Training Phase ===")
        return history, ann  # ✅ Base model is returned for inference
