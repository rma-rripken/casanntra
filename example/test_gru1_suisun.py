from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras import layers, regularizers, Model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class GRUBuilder1(ModelBuilder):

    def __init__(self, input_names, output_names, ndays):
        super().__init__(input_names, output_names)
        self.ntime = ndays
        self.ndays = ndays
        self.nwindows = 0
        self.window_length = 0
        self.reverse_time_inputs = False
        self.histories = []
        self.metrics = []  # For storing metrics

    def build_model(self, input_layers, input_data):
        prepro_layers = self.prepro_layers(input_layers, input_data)
        x = layers.Lambda(lambda x: tf.stack(x, axis=-1))(prepro_layers)

        x = layers.GRU(
            units=32, return_sequences=True, activation="sigmoid", name="gru_1"
        )(x)
        x = layers.GRU(
            units=16, return_sequences=False, activation="sigmoid", name="gru_2"
        )(x)
        x = layers.Flatten()(x)

        outdim = len(self.output_names)
        outputs = layers.Dense(
            units=outdim,
            name="ec",
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
        )(x)
        ann = Model(inputs=input_layers, outputs=outputs)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008),
            loss="mae",
            metrics=["mean_absolute_error", "mse"],
            run_eagerly=True,
        )
        return ann

    def fit_model(
        self, ann, fit_input, fit_output, test_input, test_output, nepochs=80
    ):
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=nepochs,
            batch_size=32,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True,
        )
        self.histories.append(history.history)

        # Compute predictions and metrics
        y_pred = ann.predict(test_input)
        y_true = test_output.values  # Assuming test_output is a DataFrame

        r2_scores = {}
        mse_scores = {}
        mae_scores = {}
        for idx, output_name in enumerate(self.output_names):
            y_true_i = y_true[:, idx]
            y_pred_i = y_pred[:, idx]
            r2_scores[output_name] = r2_score(y_true_i, y_pred_i)
            mse_scores[output_name] = mean_squared_error(y_true_i, y_pred_i)
            mae_scores[output_name] = mean_absolute_error(y_true_i, y_pred_i)
        # Store metrics
        self.metrics.append(
            {"r2_scores": r2_scores, "mse_scores": mse_scores, "mae_scores": mae_scores}
        )

        # Print and plot metrics
        fold_number = len(self.metrics)
        print(f"\n=== Metrics for Fold {fold_number} ===")
        print("R2 Scores:")
        for output, score in r2_scores.items():
            print(f"  {output}: {score:.4f}")
        print("MSE Scores:")
        for output, score in mse_scores.items():
            print(f"  {output}: {score:.4f}")
        print("MAE Scores:")
        for output, score in mae_scores.items():
            print(f"  {output}: {score:.4f}")

        self.plot_metrics_for_current_fold(
            r2_scores, mse_scores, mae_scores, fold_number
        )

        return history, ann

    def plot_metrics_for_current_fold(
        self, r2_scores, mse_scores, mae_scores, fold_number
    ):
        """Plots and saves metrics for the current fold."""
        metrics_dict = {"R2 Score": r2_scores, "MSE": mse_scores, "MAE": mae_scores}

        for metric_name, scores in metrics_dict.items():
            plt.figure(figsize=(10, 6))
            output_names = list(scores.keys())
            values = list(scores.values())
            plt.bar(output_names, values)
            plt.title(f"{metric_name} for Fold {fold_number}")
            plt.xlabel("Output Variables")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_filename = (
                f"output/{metric_name.replace(' ', '_').lower()}_fold_{fold_number}.png"
            )
            plt.savefig(plot_filename)
            plt.close()
            print(f"{metric_name} plot for fold {fold_number} saved to {plot_filename}")


def test_xvalid_gru():
    input_names = [
        "sac_flow",
        "exports",
        "sjr_flow",
        "cu_flow",
        "sf_tidal_energy",
        "sf_tidal_filter",
        "dcc",
        "smscg",
    ]
    output_names = [
        "x2",
        "mal",
        "nsl2",
        "bdl",
        "cse",
        "emm2",
        "tms",
        "jer",
        "sal",
        "bac",
        "oh4",
    ]
    plot_locs = ["x2", "cse", "emm2", "jer", "bdl", "sal", "bac"]
    builder = GRUBuilder1(input_names=input_names, output_names=output_names, ndays=80)

    os.makedirs("output", exist_ok=True)

    fpattern = "schism_suisun_*.csv"
    df = read_data(fpattern)

    # Adjust the number of folds here if desired
    df_in, df_out = builder.xvalid_time_folds(
        df, target_fold_len="180d", split_in_out=True
    )

    for col in output_names:
        if col == "x2":
            df_out.loc[:, col] = df_out.loc[:, col] / 100.0
        elif col in ["mal", "cse", "bdl", "nsl2"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 10000.0
        elif col in ["emm2", "jer", "tms"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 2000.0
        else:
            df_out.loc[:, col] = df_out.loc[:, col] / 1000.0

    xvalid_fit(
        df_in,
        df_out,
        builder,
        nepochs=80,
        plot_folds="all",
        plot_locs=plot_locs,
        out_prefix="output/gru1_wmetrics",
    )

    print("\n=== Analyzing Base Model Histories ===")
    plot_aggregated_histories(
        builder.histories,
        "Base Model Training Histories",
        "output/base_model_training_histories_wmetrics.png",
    )

    save_histories_to_json(
        builder.histories, "output/base_model_histories_wmetrics.json"
    )

    print("\n=== Analyzing Base Model Metrics ===")
    plot_metrics_across_folds(
        builder.metrics,
        builder.output_names,
        "Base Model Metrics",
        "output/base_model_metrics_wmetrics",
    )
    save_metrics_to_json(builder.metrics, "output/base_model_metrics_wmetrics.json")


def plot_aggregated_histories(histories, title, save_path_prefix):
    """Aggregates and plots the training and validation metrics across all folds."""
    metrics = {}
    for history in histories:
        for key, values in history.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(values)

    for metric, fold_histories in metrics.items():
        plt.figure(figsize=(10, 6))
        for fold_idx, history_values in enumerate(fold_histories):
            plt.plot(history_values, label=f"Fold {fold_idx+1}")
        plt.title(f"{title} - {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.legend()
        plt.grid(True)
        metric_save_path = f"{save_path_prefix}_{metric}.png"
        plt.savefig(metric_save_path)
        plt.close()
        print(f"{metric.capitalize()} plot saved to {metric_save_path}")


def save_histories_to_json(histories, save_path):
    """Saves all training histories to a JSON file."""
    with open(save_path, "w") as f:
        json.dump(histories, f, indent=4)
    print(f"All training histories saved to {save_path}")


def plot_metrics_across_folds(metrics_list, output_names, title, save_path_prefix):
    """Plots and saves the metrics across all folds."""
    metric_types = ["r2_scores", "mse_scores", "mae_scores"]
    for metric_name in metric_types:
        plt.figure(figsize=(12, 6))
        for output_name in output_names:
            metric_values = [
                metrics[metric_name][output_name] for metrics in metrics_list
            ]
            plt.plot(
                range(1, len(metrics_list) + 1),
                metric_values,
                marker="o",
                label=output_name,
            )
        plt.title(
            f"{title} - {metric_name.replace('_', ' ').capitalize()} Across Folds"
        )
        plt.xlabel("Fold Number")
        plt.ylabel(metric_name.replace("_", " ").capitalize())
        plt.legend()
        plt.grid(True)
        save_path = f"{save_path_prefix}_{metric_name}_across_folds.png"
        plt.savefig(save_path)
        plt.close()
        print(
            f"{metric_name.replace('_', ' ').capitalize()} across folds plot saved to {save_path}"
        )


def save_metrics_to_json(metrics_list, save_path):
    """Saves all metrics to a JSON file."""
    with open(save_path, "w") as f:
        json.dump(metrics_list, f, indent=4)
    print(f"All metrics saved to {save_path}")


if __name__ == "__main__":
    test_xvalid_gru()
