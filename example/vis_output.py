import matplotlib.pyplot as plt
import pandas as pd
import sys

# Mapping of models to output file prefixes
OUTPUT_PREFIXES = {
    "dsm2": "dsm2_base_gru2",
    "dsm2.schism": "dsm2.schism_base_gru2",
    "base.suisun": "schism_base.suisun_gru2",
    "base.suisun-secondary": "schism_base.suisun_gru2",
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
    if model == "base.suisun-secondary":
        ref_out_fname = f"{output_prefix}_xvalid_ref_out_secondary_unscaled.csv"

    print("ref", ref_out_fname)
    model_data = pd.read_csv(
        ref_out_fname, index_col=0, parse_dates=["datetime"], header=0
    )
    ann_out_fname = f"{output_prefix}_xvalid.csv"
    if model == "base.suisun-secondary":
        ann_out_fname = f"{output_prefix}_xvalid_1.csv"
    elif model == "base.suisun":
        ann_out_fname = f"{output_prefix}_xvalid_0.csv"
    ann_data = pd.read_csv(ann_out_fname, index_col=0, header=0)
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
        if i == 0:
            ax.set_title(title)
        ax.plot(
            sub_mod.datetime,
            sub_mod[station],
            linestyle=linestyle,
            label=f"{model} Model",
            color=str(0.1),
        )
        ax.plot(
            sub_ann.datetime,
            sub_ann[station],
            linestyle=linestyle,
            label=f"{model} ANN",
        )
        ax.set_ylabel("Norm EC")
        ax.set_title(f"Case = {icase}, {station}")

    axes[0].legend()


def main():
    """Main execution function to parse user input and generate plots."""
    station = sys.argv[1] if len(sys.argv) > 1 else "x2"
    models = sys.argv[2:] if len(sys.argv) > 2 else ["dsm2"]

    if len(models) > 2:
        raise ValueError("Only one or two models can be selected for visualization.")

    n_cases = 7
    fig, axes = plt.subplots(
        n_cases, sharex=False, constrained_layout=True, figsize=(8, 9)
    )

    for i, model in enumerate(models):

        model_data, ann_data = load_data(model, station)
        linestyle = "-" if i == 0 else "--"  # Solid for first, dashed for second
        plot_results(axes, model_data, ann_data, station, model, linestyle)

    plt.show()


if __name__ == "__main__":
    main()
