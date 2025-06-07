from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import sys
import os
from tqdm import tqdm
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, gridplot, row
from bokeh.models import Legend, HoverTool, ColumnDataSource, Div, Panel, Tabs
from bokeh.palettes import Category10
import traceback

# Mapping of models to output file prefixes
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
    elif "base." in model:
        ann_out_fname = f"{output_prefix}_xvalid_0.csv"
    ann_data = pd.read_csv(os.path.join("output", ann_out_fname), index_col=0, header=0)
    ann_data["datetime"] = model_data.datetime
    ann_data["case"] = model_data.case
    return model_data, ann_data


def plot_results(plots, model_data, ann_data, station, model, line_style="solid", color_index=0):
    """Plots the model vs ANN results for a given station and model using Bokeh."""
    title = STATION_NAMES.get(station, station)
    colors = Category10[10]  # Bokeh color palette

    for i, plot in enumerate(plots):
        icase = i + 1
        if model in ["rma", "dsm2.schism", "base.suisun", "base.suisun-secondary"]:
            grabcase = icase
        else:
            grabcase = icase
        print("Model: ", model, "Case being plotted: ", grabcase)

        sub_mod = model_data[model_data.case == grabcase]
        sub_ann = ann_data[ann_data.case == grabcase]

        # Create ColumnDataSource for hover tooltips
        source_model = ColumnDataSource(data={
            'datetime': sub_mod.datetime,
            'value': sub_mod[station],
            'type': [f"{model} Model"] * len(sub_mod)
        })

        source_ann = ColumnDataSource(data={
            'datetime': sub_ann.datetime,
            'value': sub_ann[station],
            'type': [f"{model} ANN"] * len(sub_ann)
        })


        model_line = plot.line(
            x='datetime',
            y='value',
            source=source_model,
            line_width=2,
            line_dash=line_style,
            color="#000000",
            legend_label=f"{model} Model"
        )

        ann_line = plot.line(
            x='datetime',
            y='value',
            source=source_ann,
            line_width=2,
            line_dash=line_style,
            color=colors[color_index],
            legend_label=f"{model} ANN"
        )

        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"

        if i == 0:
            plot.legend.visible = True
        else:
            plot.legend.visible = False

        if color_index == 0:
            # Set plot properties
            plot.title.text = f"Case = {icase}, {station}"
            plot.yaxis.axis_label = "Norm EC"

        if(color_index == 0):
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Date", "@datetime{%F}"),
                    ("Value", "@value{0.000}"),
                    ("Type", "@type")
                ],
                formatters={"@datetime": "datetime"},
                renderers=[model_line, ann_line]
            )
            plot.add_tools(hover)



def vis(models, station, output_filename=None):
    """Creates Bokeh plots for the given models and station."""
    if len(models) > 2:
        raise ValueError("Only one or two models can be selected for visualization.")

    # Get station full name
    station_name = STATION_NAMES.get(station, station)

    # Create header with station and model information
    header_text = f"<h1>Comparison for {station_name} ({station})</h1>"
    header_text += f"<h3>Models: {' vs '.join(models)}</h3>"
    header = Div(text=header_text, width=800)

    n_cases = 7
    plots = []
    tabs = []

    # Create plots for each case
    for i in range(n_cases):
        case_num = i + 1
        # Create a figure for each case
        p = figure(
            height=180,
            width=1200,
            x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            title=f"Case {case_num}"
        )
        p.toolbar.logo = None  # Disable the Bokeh logo in the toolbar

        plots.append(p)

    # Plot data for each model
    model_data_dict = {}
    for i, model in tqdm(enumerate(models)):
        model_data, ann_data = load_data(model, station)
        model_data_dict[model] = (model_data, ann_data)
        line_style = "solid" if i == 0 else "dashed"  # Solid for first, dashed for second
        plot_results(plots, model_data, ann_data, station, model, line_style, i)


    to_display = [header]
    to_display.extend(plots)

    # Create final layout with header and tabs
    layout = column(to_display)

    # Save or show the plot
    if output_filename:
        output_file(output_filename, title=f"Comparison for {station}")
        save(layout)
        print(f"Saved plot to: {output_filename}")
    else:
        # For interactive use, you would use show() here
        # But for this script, we'll always save to file
        pass


def vis_all_stations(models, subdir=None):
    """
    Create visualization plots for all stations using Bokeh.

    Args:
        models: List of model names to compare
        subdir: Subdirectory to save plots in
    """
    # Create output directory if it doesn't exist
    output_dir = "station_plots_bokeh"
    os.makedirs(output_dir, exist_ok=True)

    if subdir is not None:
        output_dir = os.path.join(output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

    # Loop through all stations
    for station_code, station_name in STATION_NAMES.items():
        print(f"Processing station: {station_name} ({station_code})")

        try:
            # Create the output filename (HTML instead of PNG)
            output_file = os.path.join(output_dir, f"comparison_{station_code}_{'_vs_'.join(models)}.html")

            # Generate and save the plot
            vis(models, station_code, output_file)

            print(f"Saved plot to: {output_file}")

        except Exception as e:
            print(f"Error processing station {station_code}: {str(e)} - traceback:{traceback.format_exc()}" )
            continue


if __name__ == "__main__":
    """Main execution function to parse user input and generate plots."""
    # station = sys.argv[1] if len(sys.argv) > 1 else "x2"
    # models = sys.argv[2:] if len(sys.argv) > 2 else ["dsm2"]

    with ProcessPoolExecutor() as executor:

        # # schism base output from the ann
        # # to
        # # schism ann
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = [f"schism_base.{compare_to}-secondary", f"schism_base.{compare_to}"]
            # vis_all_stations(models_to_compare,  f"{compare_to}/schism-ann-base_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/schism-ann-base_to_schism-ann")

        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}-secondary", f"rma_base.{compare_to}"]
            # vis_all_stations(models_to_compare, f"{compare_to}/rma-ann-base_to_rma-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann-base_to_rma-ann")

        # # schism base - not the base head of the restoration
        # # to
        # # schism restoration
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = ["dsm2.schism", f"schism_base.{compare_to}"]
            # vis_all_stations(models_to_compare, f"{compare_to}/schism-base_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/schism-base_to_schism-ann")
        #
        # # rma base (not from restoration ann)
        # # to
        # # rma restoration ann
        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = ["dsm2.rma", f"rma_base.{compare_to}"]
            # vis_all_stations(models_to_compare, f"{compare_to}/rma-base_to_rma-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-base_to_rma-ann")
        #
        # # rma ann output vs schism ann output
        for compare_to in ["suisun", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}", f"schism_base.{compare_to}"]
            # vis_all_stations(models_to_compare, f"{compare_to}/rma-ann_to_schism-ann")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann_to_schism-ann")
        #
        for compare_to in ["suisun", "cache", 'ft']:
            models_to_compare = ["dsm2.rma", f"rma_base.{compare_to}-secondary"]
            # vis_all_stations(models_to_compare, f"{compare_to}/rma-base_to_rma-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-base_to_rma-ann-base")

        # # rma-ann-base to schism-ann-base
        for compare_to in ["suisun", 'ft']:
            models_to_compare = [f"rma_base.{compare_to}-secondary", f"schism_base.{compare_to}-secondary"]
            # vis_all_stations(models_to_compare, f"{compare_to}/rma-ann-base_to_schism-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/rma-ann-base_to_schism-ann-base")
        #
        # # schism-base to schism-ann-base
        for compare_to in ["suisun", "ft", "slr"]:
            models_to_compare = ["dsm2.schism", f"schism_base.{compare_to}-secondary"]
            # vis_all_stations(models_to_compare, f"{compare_to}/schism-base_to_schism-ann-base")
            executor.submit(vis_all_stations, models_to_compare, f"{compare_to}/schism-base_to_schism-ann-base")

