# Reduce regularization
# Larger model
# MSE or weighted MAE
# Explicit interactions
# Augment cases
# transform


import pandas as pd
from tensorflow.keras.layers import Lambda
from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi
from casanntra.tide_transforms import *


def modified_exponential_decay(x, a=5e-5, b=80000):
    exp_decay = tf.exp(-a * x)
    exp_decay_tapered = (exp_decay - tf.exp(-a * b)) / (1 - tf.exp(-a * b))
    return exp_decay_tapered


class GRUBuilder2m(ModelBuilder):

    def __init__(self, input_names, output_names, ndays):
        super().__init__(input_names, output_names)
        self.ntime = ndays
        self.ndays = ndays
        self.nwindows = 0
        self.window_length = 0
        self.reverse_time_inputs = False

    def prepro_layers(self, inp_layers, df):
        """Examples of processing dataframe with multiindex of locations and lags into normalized inputs"""
        layers = []
        names = self.feature_names()
        if len(names) != len(inp_layers):
            raise ValueError(
                "Inconsistency in number of layers between inp_layers and feature names"
            )
        thresh = 40000.0
        dims = {x: self.feature_dim(x) for x in names}

        for fndx, feature in enumerate(self.feature_names()):
            station_df = df.loc[:, feature]
            xinput = inp_layers[fndx]
            prepro_name = f"{feature}_prepro"
            if feature in ["dcc", "smscg"] and False:
                feature_layer = Normalization(
                    axis=None, name=prepro_name
                )  # Rescaling(1.0)
            elif feature in ["sac_flow", "ndo"] and thresh is not None:
                # scale_factor = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32)
                feature_layer = Lambda(
                    lambda x: modified_exponential_decay(x), name=prepro_name
                )
                # feature_layer = trainable_scale(Rescaling(1 / thresh, name=prepro_name)  # Normalization(axis=None)
            elif feature == "sjr_flow" and thresh is not None:
                feature_layer = Rescaling(
                    0.25 / thresh, name=prepro_name
                )  # Normalization(axis=None)
            else:
                feature_layer = Normalization(axis=None, name=prepro_name)
                feature_layer.adapt(station_df.to_numpy())
            layers.append(feature_layer(xinput))
        return layers

    def build_model(self, input_layers, input_data):

        prepro_layers = self.prepro_layers(input_layers, input_data)
        x = layers.Lambda(lambda x: tf.stack(x, axis=-1))(prepro_layers)
        x = layers.GRU(
            units=8, return_sequences=True, activation="sigmoid", name="gru_1"
        )(x)
        x = layers.GRU(
            units=8, return_sequences=False, activation="sigmoid", name="gru_2"
        )(x)
        x = layers.Flatten()(x)

        outdim = len(self.output_names)
        # The regularization is unknown
        outputs = layers.Dense(
            units=outdim,
            name="ec",
            activation="elu",
            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001),
        )(x)
        ann = Model(inputs=input_layers, outputs=outputs)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008),
            loss="mse",  # could be mean_absolute_error or mean_squared_error
            metrics=["mean_absolute_error", "mse"],
            run_eagerly=False,
        )
        return ann

    def fit_model(
        self, ann, fit_input, fit_output, test_input, test_output, nepochs=80
    ):  # ,tcb):
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008),
            loss="mae",  # could be mean_absolute_error or mean_squared_error
            metrics=["mean_absolute_error", "mse"],
            run_eagerly=False,
        )
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=10,
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True,
        )
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
            loss="mae",  # could be mean_absolute_error or mean_squared_error
            metrics=["mean_absolute_error", "mse"],
            run_eagerly=False,
        )
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=(nepochs - 10),
            batch_size=64,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True,
        )
        return history, ann


def test_gru_multi():
    # todo: note derivative
    input_names = ["ndo", "tidal_pc1", "tidal_pc2", "smscg"]
    output_names = ["x2", "mal", "god", "vol", "bdl", "nsl2", "cse", "gzl"]
    plot_locs = ["x2", "cse", "god", "bdl", "gzl", "cse"]
    builder = GRUBuilder2m(input_names=input_names, output_names=output_names, ndays=80)

    fpattern = "dsm2_base_*.csv"
    df = read_data(fpattern)
    df = append_tidal_pca_cols(df)

    df_in, df_out = builder.xvalid_time_folds(
        df, target_fold_len="180d", split_in_out=True
    )  # adds a column called 'fold'

    df_in["fold"] = df_in["fold"] % 10
    df_out["fold"] = df_out["fold"] % 10

    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:, col] = df_out.loc[:, col] / 100.0
        elif col in ["mrz", "pct", "mal", "gzl", "god", "vol", "cse", "bdl", "nsl2"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 12000.0
        elif col in ["jer", "frk"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 2500.0
        elif col in ["emm2", "tms", "anh"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 3000.0
        else:
            df_out.loc[:, col] = df_out.loc[:, col] / 1500.0

    # xvalid_fit(df_in,df_out,builder,plot_folds=[0,1],plot_locs=plot_locs)
    xvalid_fit_multi(
        df_in,
        df_out,
        builder,
        plot_folds="all",
        plot_locs=plot_locs,
        out_prefix="output/dsm2_gru2.pc2,",
        nepochs=100,
        pool_size=11,
    )


if __name__ == "__main__":
    test_gru_multi()
