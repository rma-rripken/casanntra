from keras.models import load_model
from tensorflow.keras import layers, regularizers, Model
import tensorflow as tf
from casanntra.model_builder import GRUBuilder2




def weighted_mtl_loss(y_true, y_pred, both_scenarios_exist=True):
    """Handles missing scenario data by computing absolute MAE when necessary."""
    loss_abs = tf.keras.losses.MAE(y_true[0], y_pred[0])  # Absolute accuracy loss

    if both_scenarios_exist:
        loss_diff = tf.keras.losses.MSE(y_true[1], y_pred[1])  # Difference loss
        return loss_abs + 0.1 * loss_diff
    else:
        return loss_abs  # If no difference can be computed, fall back to absolute MAE

def contrastive_loss(y_true, y_pred):
    """Contrastive loss enforcing separation between scenarios."""
    base, suisun = y_pred[:, 0], y_pred[:, 1]
    margin = 0.1  # Controls scenario separation strength
    return tf.reduce_mean(tf.maximum(base - suisun + margin, 0.0))



# multi_stage_model_builder.py
class MultiStageModelBuilder(GRUBuilder2):

    def __init__(self, input_names, output_names, ndays=90, **builder_args):
        """Multi-stage model builder that supports flexible transfer learning options."""
        super().__init__(input_names, output_names, ndays)

        self.builder_args = builder_args  # ✅ Store all arguments for reference
        self.transfer_type = builder_args.get("transfer_type", None)  # ✅ Default to None if missing


        # ✅ Register additional loss functions required for staged training
        self.register_custom_object("contrastive_loss", contrastive_loss)
        self.register_custom_object("weighted_mtl_loss", weighted_mtl_loss)

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

        # ✅ Explicitly create the final output layer
        scaled_output = layers.Dense(len(self.output_names), activation='linear', name="scaled_output")(feature_extractor)

        # ✅ Apply unscaling only if requested
        if add_unscaled_output:
            unscaled_output = self._create_unscaled_layer(scaled_output)
            return Model(inputs=input_layer, outputs=[scaled_output, unscaled_output])

        return Model(inputs=input_layer, outputs=scaled_output)


    def _build_base_model(self, input_layer, feature_extractor):
        """Creates a model for original training with only absolute salinity prediction."""
        ec_absolute = layers.Dense(
            units=len(self.output_names), name="ec_absolute", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)
        return Model(inputs=input_layer, outputs=ec_absolute)

    def _build_mtl_model(self, input_layer, feature_extractor):
        """Creates the MTL architecture with absolute and scenario difference outputs."""
        ec_absolute = layers.Dense(
            units=len(self.output_names), name="ec_absolute", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        ec_diff = layers.Dense(
            units=len(self.output_names), name="ec_diff", activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        return Model(inputs=input_layer, outputs=[ec_absolute, ec_diff])

    def _build_contrastive_model(self, input_layer, feature_extractor):
        """Creates a contrastive learning model with scenario separation."""
        ec_base = layers.Dense(
            units=len(self.output_names), name="ec_base", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        ec_suisun = layers.Dense(
            units=len(self.output_names), name="ec_suisun", activation='elu',
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(feature_extractor)

        combined_output = layers.Concatenate(name="combined_output")([ec_base, ec_suisun])

        return Model(inputs=input_layer, outputs=[ec_base, ec_suisun, combined_output])


    def fit_model(self,
                ann,
                fit_input,
                fit_output,
                test_input,
                test_output,
                init_train_rate,
                init_epochs,
                main_train_rate,
                main_epochs):  
        """Custom fit_model for MultiStageModelBuilder to support staged learning."""

        loss_function = "mae"

        # ✅ Initial training phase (faster learning rate)
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=init_train_rate, clipnorm=1.0), 
            loss=loss_function,
            metrics=['mae', 'mse'],
            run_eagerly=True
        )
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=init_epochs,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
        )

        # ✅ Main training phase (slower learning rate)
        if main_epochs is not None and main_epochs > 0:
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=main_train_rate), 
                loss=loss_function,
                metrics=['mae', 'mse'],
                run_eagerly=False
            )
            history = ann.fit(
                fit_input,
                fit_output,
                epochs=main_epochs,
                batch_size=64,
                validation_data=(test_input, test_output),
                verbose=2,
                shuffle=True
            )

        return history, ann