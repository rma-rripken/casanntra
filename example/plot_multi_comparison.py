import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



OUT_FOLDER = "station_bar_plots"  

TRIALS = [
    {
        "name": "Direct",
        "type": "direct",
        "base_prefix": "dsm2.schism_base_gru2",
        "base_suffix": "_d",
        "suisun_prefix": "schism_base.suisun_gru2",
        "suisun_suffix": "_d",
        "contrast_weight": None
    },
    {
        "name": "Contrast_Base",
        "type": "contrastive",
        "contrast_prefix": "schism_base.suisun_gru2",
        "contrast_suffix": "",  
        "contrast_weight": 0.5
    },
    {
        "name": "Contrast_Trial8",
        "type": "contrastive",
        "contrast_prefix": "schism_base.suisun_gru2",
        "contrast_suffix": "_Trial8",
        "contrast_weight": 1.0
    },
    {
        "name": "Contrast_Trial66",
        "type": "contrastive",
        "contrast_prefix": "schism_base.suisun_gru2",
        "contrast_suffix": "_Trial66",
        "contrast_weight": 1.0
    }
]

STATIONS = [
    "x2","pct","mal","god","vol","bdl","nsl2","cse","emm2","tms",
    "anh","jer","gzl","sal","frk","bac","rsl","oh4","trp"
]

def compute_mse(true_vals, pred_vals):
    mask = (~np.isnan(true_vals)) & (~np.isnan(pred_vals))
    if mask.sum() == 0:
        return np.nan
    diff = true_vals[mask] - pred_vals[mask]
    return np.mean(diff**2)

def merge_ref_and_pred(pred_file, ref_file):
    if not os.path.exists(pred_file) or not os.path.exists(ref_file):
        print(f"[merge_ref_and_pred] Missing => {pred_file} or {ref_file}")
        return None

    df_pred = pd.read_csv(pred_file, parse_dates=["datetime"])
    df_ref  = pd.read_csv(ref_file, parse_dates=["datetime"])
    df_merged = pd.merge(df_ref, df_pred, on=["datetime","case"], how="inner",
                         suffixes=("", "_pred"))
    return df_merged


def load_direct_approach(base_prefix, base_suffix, suisun_prefix, suisun_suffix, station_list):
    bpfile = f"{base_prefix}{base_suffix}_xvalid.csv"
    brfile = f"{base_prefix}{base_suffix}_xvalid_ref_out_unscaled.csv"
    spfile = f"{suisun_prefix}{suisun_suffix}_xvalid.csv"
    srfile = f"{suisun_prefix}{suisun_suffix}_xvalid_ref_out_unscaled.csv"

    df_base = merge_ref_and_pred(bpfile, brfile)
    df_suisun = merge_ref_and_pred(spfile, srfile)

    if df_base is None or df_suisun is None:
        return {st: (np.nan, np.nan, np.nan) for st in station_list}

    base_rename = {}
    for stn in station_list:
        base_rename[stn] = f"{stn}_base"
        base_rename[f"{stn}_pred"] = f"{stn}_base_pred"
    df_base = df_base.rename(columns=base_rename)

    suisun_rename = {}
    for stn in station_list:
        suisun_rename[stn] = f"{stn}_suisun"
        suisun_rename[f"{stn}_pred"] = f"{stn}_suisun_pred"
    df_suisun = df_suisun.rename(columns=suisun_rename)

    df_merged = pd.merge(df_base, df_suisun, on=["datetime","case"], how="inner")

    station_errors = {}
    for stn in station_list:
        btrue_col = f"{stn}_base"
        bpred_col = f"{stn}_base_pred"
        strue_col = f"{stn}_suisun"
        spred_col = f"{stn}_suisun_pred"
        if btrue_col not in df_merged.columns or strue_col not in df_merged.columns:
            station_errors[stn] = (np.nan, np.nan, np.nan)
            continue

        base_true   = df_merged[btrue_col].values
        base_pred   = df_merged[bpred_col].values
        suisun_true = df_merged[strue_col].values
        suisun_pred = df_merged[spred_col].values

        base_mse   = compute_mse(base_true, base_pred)
        suisun_mse = compute_mse(suisun_true, suisun_pred)
        diff_mse   = compute_mse((suisun_true - base_true), (suisun_pred - base_pred))

        station_errors[stn] = (base_mse, suisun_mse, diff_mse)

    return station_errors

def load_contrastive_approach(contrast_prefix, contrast_suffix, station_list, contrast_weight=None):
    tgt_pred_file = f"{contrast_prefix}{contrast_suffix}_xvalid_0.csv"
    tgt_ref_file  = f"{contrast_prefix}{contrast_suffix}_xvalid_ref_out_unscaled.csv"
    src_pred_file = f"{contrast_prefix}{contrast_suffix}_xvalid_1.csv"
    src_ref_file  = f"{contrast_prefix}{contrast_suffix}_xvalid_ref_out_secondary_unscaled.csv"

    df_target = merge_ref_and_pred(tgt_pred_file, tgt_ref_file)
    df_source = merge_ref_and_pred(src_pred_file, src_ref_file)
    if df_target is None or df_source is None:
        return {st: (np.nan, np.nan, np.nan) for st in station_list}

    tgt_rename = {}
    for stn in station_list:
        tgt_rename[stn] = f"{stn}_tgt"
        tgt_rename[f"{stn}_pred"] = f"{stn}_tgt_pred"
    df_target = df_target.rename(columns=tgt_rename)

    src_rename = {}
    for stn in station_list:
        src_rename[stn] = f"{stn}_src"
        src_rename[f"{stn}_pred"] = f"{stn}_src_pred"
    df_source = df_source.rename(columns=src_rename)

    df_merged = pd.merge(df_target, df_source, on=["datetime","case"], how="inner")

    station_errors = {}
    for stn in station_list:
        tgt_true_col = f"{stn}_tgt"
        tgt_pred_col = f"{stn}_tgt_pred"
        src_true_col = f"{stn}_src"
        src_pred_col = f"{stn}_src_pred"
        if tgt_true_col not in df_merged.columns or src_true_col not in df_merged.columns:
            station_errors[stn] = (np.nan, np.nan, np.nan)
            continue

        tgt_true = df_merged[tgt_true_col].values
        tgt_pred = df_merged[tgt_pred_col].values
        src_true = df_merged[src_true_col].values
        src_pred = df_merged[src_pred_col].values

        base_mse   = compute_mse(src_true, src_pred)
        suisun_mse = compute_mse(tgt_true, tgt_pred)
        diff_mse   = compute_mse((tgt_true - src_true), (tgt_pred - src_pred))

        station_errors[stn] = (base_mse, suisun_mse, diff_mse)

    return station_errors


def plot_station_comparison(station, approach_data, out_folder):
    """
    For a single station, we have multiple approaches, each providing:
      - base_mse
      - suisun_mse
      - diff_mse
    We'll produce a bar chart with X-axis => approach name (optionally including (λ=...) if relevant),
    3 bars => base, suisun, diff, each a different color.
    Save as PNG in out_folder / f"{station}.png"
    """

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    display_names = []
    base_vals   = []
    suisun_vals = []
    diff_vals   = []

    for app in approach_data:
        if app["contrast_weight"] is not None:
            label = f"{app['name']} (λ={app['contrast_weight']})"
        else:
            label = app["name"]

        display_names.append(label)

        (bm, sm, dm) = app["errors"].get(station, (np.nan, np.nan, np.nan))
        base_vals.append(bm)
        suisun_vals.append(sm)
        diff_vals.append(dm)

    x_positions = np.arange(len(display_names))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, len(display_names)*1.2), 4))

    # Plot the 3 bars side by side
    ax.bar(x_positions - bar_width, base_vals, 
           width=bar_width, color="C0", alpha=0.8, label="Base Error")
    ax.bar(x_positions, suisun_vals, 
           width=bar_width, color="C1", alpha=0.8, label="Suisun Error")
    ax.bar(x_positions + bar_width, diff_vals,  
           width=bar_width, color="C2", alpha=0.8, label="Diff Error")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_names, rotation=15, ha="right")

    ax.set_ylabel("MSE")
    ax.set_title(f"Station: {station}")
    ax.legend()

    out_png = os.path.join(out_folder, f"{station}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_station_comparison] => saved {out_png}")

def main():
    approach_data = []
    for trial in TRIALS:
        a_type = trial["type"]
        name = trial["name"]
        if a_type == "direct":
            base_px = trial["base_prefix"]
            base_sfx = trial["base_suffix"]
            suisun_px = trial["suisun_prefix"]
            suisun_sfx= trial["suisun_suffix"]
            errs = load_direct_approach(base_px, base_sfx, suisun_px, suisun_sfx, STATIONS)
            approach_data.append({
                "name": name,
                "errors": errs,
                "contrast_weight": trial.get("contrast_weight", None)
            })
        elif a_type == "contrastive":
            c_px = trial["contrast_prefix"]
            c_sfx = trial["contrast_suffix"]
            errs = load_contrastive_approach(c_px, c_sfx, STATIONS, trial.get("contrast_weight", None))
            approach_data.append({
                "name": name,
                "errors": errs,
                "contrast_weight": trial.get("contrast_weight", None)
            })
        else:
            print(f"[WARN] Unknown approach type: {a_type}")

    for stn in STATIONS:
        plot_station_comparison(stn, approach_data, OUT_FOLDER)

if __name__ == "__main__":
    main()