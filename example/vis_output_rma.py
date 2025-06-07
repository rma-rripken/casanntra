from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from tqdm import tqdm

# Mapping of models to output file prefixes
# the -secondary are the ann-base
# the ones without -secondary are the ann

OUTPUT_PREFIXES = {
    "dsm2": "dsm2_base_gru2",
    "dsm2.schism": "dsm2.schism_base_gru2",
    "schism_base.suisun": "schism_base.suisun_gru2",
    "schism_base.suisun-secondary": "schism_base.suisun_gru2",
    "schism_base.slr": "schism_base.slr_gru2",
    "schism_base.slr-secondary": "schism_base.slr_gru2",
    "schism_base.ft": "schism_base.ft_gru2",
    "schism_base.ft-secondary": "schism_base.ft_gru2",
    "dsm2.rma": "dsm2.rma_base_gru2",
    "rma_base.suisun": "rma_base.suisun_gru2",
    "rma_base.suisun-secondary": "rma_base.suisun_gru2",
    "rma_base.cache": "rma_base.cache_gru2",
    "rma_base.cache-secondary": "rma_base.cache_gru2",
    "rma_base.ft": "rma_base.ft_gru2",
    "rma_base.ft-secondary": "rma_base.ft_gru2",
}

LABELS = {
    "dsm2.schism": "schism base",
    "schism_base.suisun": "schism ann suisun",
    "schism_base.suisun-secondary": "schism ann base",
    "schism_base.slr": "schism ann slr",
    "schism_base.slr-secondary": "schism ann base",
    "schism_base.ft": "schism ann ft",
    "schism_base.ft-secondary": "schism ann base",
    "dsm2.rma": "rma base",
    "rma_base.suisun": "rma ann suisun",
    "rma_base.suisun-secondary": "rma ann base",
    "rma_base.cache": "rma ann cache",
    "rma_base.cache-secondary": "rma ann base",
    "rma_base.ft": "rma ann ft",
    "rma_base.ft-secondary": "rma ann base",
}


# Mapping of station short names to full names
STATION_NAMES = {
    "cse": "Collinsville",
    "rsl": "Rock Slough",
    "oh4": "Old@HW4",
    "frk": "Franks Tract",
    "bac": "Old R. at Bacon",
    "x2": "X2",
    "emm2": "Emmaton",
    "jer": "Jersey Point",
    "bdl": "Beldon's Landing",
    "mal": "Mallard",
    "hll": "Holland Cut",
    "sal": "San Andreas Landing",
    "god": "Godfather Slough",
    "gzl": "Grizzly Bay",
    "vol": "Suisun Sl at Volanti",
}


def load_data(model: str, station: str):
    """Loads model and ANN data for the given model and station."""
    output_prefix = OUTPUT_PREFIXES.get(model, None)
    if output_prefix is None:
        raise ValueError(f"Invalid model: {model}")

    ref_out_fname = f"{output_prefix}_xvalid_ref_out_unscaled.csv"
    if "-secondary" in model:
        ref_out_fname = f"{output_prefix}_xvalid_ref_out_secondary_unscaled.csv"

    print("ref", ref_out_fname)
    model_data = pd.read_csv(os.path.join("output", ref_out_fname),
         index_col=0, parse_dates=["datetime"], header=0
    )
    ann_out_fname = f"{output_prefix}_xvalid.csv"
    if "-secondary" in model:
        ann_out_fname = f"{output_prefix}_xvalid_1.csv"
    elif  "base." in model:
        ann_out_fname = f"{output_prefix}_xvalid_0.csv"
    ann_data = pd.read_csv(os.path.join("output", ann_out_fname), index_col=0, header=0)
    ann_data["datetime"] = model_data.datetime
    ann_data["case"] = model_data.case
    return model_data, ann_data


def plot_results(axes, model_data, ann_data, station, model, linestyle="-"):
    """Plots the model vs ANN results for a given station and model on the same figure."""
    title = STATION_NAMES.get(station, station)

    for i, ax in enumerate(axes):
        icase = i + 1
        # icase = i + 1 +i*10 + 11
        if model in ["rma", "dsm2.schism", "base.suisun", "base.suisun-secondary"]:
            grabcase = icase
        else:
            grabcase = 1000 + icase
        grabcase = icase
        print("Model: ", model, "Case being plotted: ", grabcase)
        sub_mod = model_data[model_data.case == grabcase]
        sub_ann = ann_data[ann_data.case == grabcase]

        model_label = LABELS.get(model, model)
        if i == 0:
            ax.set_title(title)
        ax.plot(
            sub_mod.datetime,
            sub_mod[station],
            linestyle=linestyle,
            label=f"{model_label} Model",
            color=str(0.1),
        )
        ax.plot(
            sub_ann.datetime,
            sub_ann[station],
            linestyle=linestyle,
            label=f"{model_label} ANN",
        )
        ax.set_ylabel("Norm EC")
        ax.set_title(f"Case = {icase}, {station}")

    axes[0].legend()


def vis(models, station,  output_filename=None):
    if len(models) > 2:
        raise ValueError("Only one or two models can be selected for visualization.")
    n_cases = 7
    fig, axes = plt.subplots(
        n_cases, sharex=False, constrained_layout=True, figsize=(8, 9)
    )
    for i, model in tqdm(enumerate(models)):
        model_data, ann_data = load_data(model, station)
        linestyle = "-" if i == 0 else "--"  # Solid for first, dashed for second
        plot_results(axes, model_data, ann_data, station, model, linestyle)
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def vis_all_stations(models, subdir=None):
    """
    Create visualization plots for all stations.

    Args:
        models: List of model names to compare
    """
    # Create output directory if it doesn't exist
    output_dir = "station_plots"
    os.makedirs(output_dir, exist_ok=True)

    if subdir is not None:
        output_dir = os.path.join(output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

    # Loop through all stations
    for station_code, station_name in STATION_NAMES.items():
        print(f"Processing station: {station_name} ({station_code})")

        try:
            # Create the output filename
            output_file = os.path.join(output_dir, f"comparison_{station_code}_{'_vs_'.join(models)}.png")

            # Generate and save the plot
            vis(models, station_code, output_file)

            print(f"Saved plot to: {output_file}")

        except Exception as e:
            print(f"Error processing station {station_code}: {str(e)}")
            continue


if __name__ == "__main__":
    """Main execution function to parse user input and generate plots."""


    # There are 6 "things" being compared here.
    # rma-base - this is model trained on dsm2 then "direct" trained on rma_base model output
    # rma-ann - this takes rma-base and constrastive trains on rma_scenario output. This is the restoration head
    # rma-ann-base same ann as rma-ann but this is the other head that returns the "base" prediction
    # schism-base
    # schism-ann
    # schism-ann-base

    # the -secondary are the ann-base
    # the ones without -secondary are the ann

    with ProcessPoolExecutor() as executor:

        # schism base output from the ann
        # to
        # schism ann
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = [ f"schism_base.{compare_to}-secondary", f"schism_base.{compare_to}"]
            #vis_all_stations(models_to_compare,  f"{compare_to}/schism-ann-base_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare,  f"{compare_to}/schism-ann-base_to_schism-ann")


        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}-secondary", f"rma_base.{compare_to}" ]
            #vis_all_stations(models_to_compare, f"{compare_to}/rma-ann-base_to_rma-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann-base_to_rma-ann")

        # schism base - not the base head of the restoration
        # to
        # schism restoration
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = ["dsm2.schism", f"schism_base.{compare_to}"]
            #vis_all_stations(models_to_compare, f"{compare_to}/schism-base_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/schism-base_to_schism-ann")

        # rma base (not from restoration ann)
        # to
        # rma restoration ann
        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = ["dsm2.rma", f"rma_base.{compare_to}"]
            #vis_all_stations(models_to_compare, f"{compare_to}/rma-base_to_rma-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-base_to_rma-ann")

        # rma ann output vs schism ann output
        for compare_to in ["suisun", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}", f"schism_base.{compare_to}"]
            #vis_all_stations(models_to_compare, f"{compare_to}/rma-ann_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann_to_schism-ann")


        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = ["dsm2.rma", f"rma_base.{compare_to}-secondary"]
            #vis_all_stations(models_to_compare, f"{compare_to}/rma-base_to_rma-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-base_to_rma-ann-base")

        # rma-ann-base to schism-ann-base
        for compare_to in ["suisun", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}-secondary", f"schism_base.{compare_to}-secondary"]
            #vis_all_stations(models_to_compare, f"{compare_to}/rma-ann-base_to_schism-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann-base_to_schism-ann-base")

        # schism-base to schism-ann-base
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = ["dsm2.schism", f"schism_base.{compare_to}-secondary"]
            #vis_all_stations(models_to_compare, f"{compare_to}/schism-base_to_schism-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/schism-base_to_schism-ann-base")

