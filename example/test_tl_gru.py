from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit
from test_gru1 import GRUBuilder1  # Ensure this import path is correct

import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from tensorflow.keras import layers, regularizers, Model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class TransferLearningGRUBuilder1(GRUBuilder1):
    def __init__(self, input_names, output_names, ndays, base_model_weights=None):
        super().__init__(input_names, output_names, ndays)
        self.base_model_weights = base_model_weights
        self.histories = []
        self.metrics = []  # For storing metrics

    def build_model(self, input_layers, input_data):
        ann = super().build_model(input_layers, input_data)
        if self.base_model_weights is not None:
            ann.load_weights(self.base_model_weights)
            ann.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
                loss='mae',
                metrics=['mean_absolute_error', 'mse'],
                run_eagerly=True
            )
        return ann

    def fit_model(self, ann, fit_input, fit_output, test_input, test_output, nepochs=30):
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=nepochs,
            batch_size=32,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
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
        self.metrics.append({
            'r2_scores': r2_scores,
            'mse_scores': mse_scores,
            'mae_scores': mae_scores
        })

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

        self.plot_metrics_for_current_fold(r2_scores, mse_scores, mae_scores, fold_number)

        return history, ann

    def plot_metrics_for_current_fold(self, r2_scores, mse_scores, mae_scores, fold_number):
        """Plots and saves metrics for the current fold."""
        metrics_dict = {
            'R2 Score': r2_scores,
            'MSE': mse_scores,
            'MAE': mae_scores
        }

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
            plot_filename = f"output/transfer_{metric_name.replace(' ', '_').lower()}_fold_{fold_number}_wmetrics.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"{metric_name} plot for fold {fold_number} saved to {plot_filename}")

def test_transfer_learning_gru1():
    input_names = ["sac_flow", "exports", "sjr_flow", "cu_flow",
                   "sf_tidal_energy", "sf_tidal_filter", "dcc", "smscg"]
    output_names = ["x2", "mal", "nsl2", "bdl", "cse", "emm2", "tms",
                    "jer", "sal", "bac", "oh4"]
    plot_locs = ["x2", "cse", "emm2", "jer", "bdl", "sal", "bac"]

    os.makedirs("output", exist_ok=True)

    print("=== Training Base Model ===")
    base_pattern = "schism_base_*.csv"
    base_df = read_data(base_pattern)
    df_in_base = base_df[["datetime", "case"] + input_names]
    df_out_base = base_df[["datetime", "case"] + output_names]

    # Normalize outputs
    for col in output_names:
        if col == "x2":
            df_out_base[col] = df_out_base[col] / 100.
        elif col in ["mal", "cse", "bdl", "nsl2"]:
            df_out_base[col] = df_out_base[col] / 10000.
        elif col in ["emm2", "jer", "tms"]:
            df_out_base[col] = df_out_base[col] / 2000.
        else:
            df_out_base[col] = df_out_base[col] / 1000.

    builder_base = GRUBuilder1(input_names=input_names, output_names=output_names, ndays=80)

    input_layers = builder_base.input_layers()
    model_base = builder_base.build_model(input_layers, df_in_base)

    df_lag_in_base = builder_base.calc_antecedent_preserve_cases(df_in_base)
    df_out_aligned_base = df_out_base.reindex(df_lag_in_base.index)
    idx = pd.IndexSlice
    fit_in_base = builder_base.df_by_feature_and_time(df_lag_in_base)
    fit_in_base = {
        name: fit_in_base.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder_base.input_names
    }

    history_base, model_base = builder_base.fit_model(
        model_base,
        fit_in_base,
        df_out_aligned_base[output_names],
        fit_in_base,
        df_out_aligned_base[output_names],
        nepochs=80
    )

    model_weights_path = "output/transfer_gru1_base_weights_wmetrics.h5"
    model_base.save_weights(model_weights_path)
    print(f"Base model weights saved to {model_weights_path}")

    plot_training_history(history_base.history, "Transfer Learning Base Model Training History", "output/transfer_base_model_history_wmetrics.png")

    print("\n=== Performing Transfer Learning ===")
    fpattern = "schism_suisun_*.csv"
    df_suisun = read_data(fpattern)
    df_in_suisun, df_out_suisun = builder_base.xvalid_time_folds(
        df_suisun, target_fold_len='180d', split_in_out=True
    )

    # Normalize outputs
    for col in output_names:
        if col == "x2":
            df_out_suisun[col] = df_out_suisun[col] / 100.
        elif col in ["mal", "cse", "bdl", "nsl2"]:
            df_out_suisun[col] = df_out_suisun[col] / 10000.
        elif col in ["emm2", "jer", "tms"]:
            df_out_suisun[col] = df_out_suisun[col] / 2000.
        else:
            df_out_suisun[col] = df_out_suisun[col] / 1000.

    builder_transfer = TransferLearningGRUBuilder1(
        input_names=input_names,
        output_names=output_names,
        ndays=80,
        base_model_weights=model_weights_path
    )

    xvalid_fit(
        df_in_suisun,
        df_out_suisun,
        builder_transfer,
        nepochs=30,
        plot_folds="all",
        plot_locs=plot_locs,
        out_prefix="output/gru1_transfer_wmetrics"
    )

    print("\n=== Analyzing Transfer Learning Histories ===")
    plot_aggregated_histories(
        builder_transfer.histories,
        "Transfer Learning Training Histories",
        "output/transfer_learning_histories_wmetrics"
    )
    save_histories_to_json(
        builder_transfer.histories,
        "output/transfer_learning_histories_wmetrics.json"
    )

    print("\n=== Analyzing Transfer Learning Metrics ===")
    plot_metrics_across_folds(
        builder_transfer.metrics,
        builder_transfer.output_names,
        "Transfer Learning Metrics",
        "output/transfer_learning_metrics_wmetrics"
    )
    save_metrics_to_json(
        builder_transfer.metrics,
        "output/transfer_learning_metrics_wmetrics.json"
    )

def plot_training_history(history, title, save_path):
    """Plots and saves the training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

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
            plt.plot(history_values, label=f'Fold {fold_idx+1}')
        plt.title(f'{title} - {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.legend()
        plt.grid(True)
        metric_save_path = f"{save_path_prefix}_{metric}.png"
        plt.savefig(metric_save_path)
        plt.close()
        print(f"{metric.capitalize()} plot saved to {metric_save_path}")

def save_histories_to_json(histories, save_path):
    """Saves all training histories to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(histories, f, indent=4)
    print(f"All training histories saved to {save_path}")

def plot_metrics_across_folds(metrics_list, output_names, title, save_path_prefix):
    """Plots and saves the metrics across all folds."""
    metric_types = ['r2_scores', 'mse_scores', 'mae_scores']
    for metric_name in metric_types:
        plt.figure(figsize=(12, 6))
        for output_name in output_names:
            metric_values = [metrics[metric_name][output_name] for metrics in metrics_list]
            plt.plot(range(1, len(metrics_list) + 1), metric_values, marker='o', label=output_name)
        plt.title(f"{title} - {metric_name.replace('_', ' ').capitalize()} Across Folds")
        plt.xlabel("Fold Number")
        plt.ylabel(metric_name.replace('_', ' ').capitalize())
        plt.legend()
        plt.grid(True)
        save_path = f"{save_path_prefix}_{metric_name}_across_folds.png"
        plt.savefig(save_path)
        plt.close()
        print(f"{metric_name.replace('_', ' ').capitalize()} across folds plot saved to {save_path}")

def save_metrics_to_json(metrics_list, save_path):
    """Saves all metrics to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    print(f"All metrics saved to {save_path}")

if __name__ == "__main__":
    test_transfer_learning_gru1()