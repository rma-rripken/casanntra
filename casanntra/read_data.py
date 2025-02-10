import pandas as pd
import os
import re
import glob

data_repo = os.path.split(__file__)[0]+"/../data"

def filter_files(fnames,masks,verbose=False):
    if masks is None or masks == "None": 
        return fnames
    if isinstance(masks,str):
        masks = [masks]
    mask_res=[re.compile(x) for x in masks]

    def exclude(mask_res,f):
        masked = any([mr.match(os.path.split(f)[1]) for mr in mask_res])
        if verbose:
            print(f"file {f}. Masking = {masked}")
        return masked
    return [f for f in fnames if not exclude(mask_res, f)]


def compute_scenario_differences(df, base_col, suisun_col):
    """
    Computes (schism, suisun) - (schism, base) differences.
    If a case is missing in one scenario, it falls back to using absolute MAE.
    """
    df = df.copy()
    df['scenario_difference'] = None  # Initialize new column

    available_cases = df.groupby("case")

    for case, group in available_cases:
        base_exists = base_col in group.columns
        suisun_exists = suisun_col in group.columns

        if base_exists and suisun_exists:
            df.loc[group.index, 'scenario_difference'] = group[suisun_col] - group[base_col]
        elif base_exists:
            df.loc[group.index, 'scenario_difference'] = group[base_col]  # Fall back to absolute MAE
        elif suisun_exists:
            df.loc[group.index, 'scenario_difference'] = group[suisun_col]  # Fall back to absolute MAE

    return df


def read_data(file_pattern,input_mask_regex,data_dir = data_repo):
    """ Read data
        file_pattern is a pattern for matching
        input_mask is a list of masks to use to filter out excluded cases
        
    """
    dss = []
    file_pattern = os.path.join(data_dir,file_pattern)
    fnames = glob.glob(file_pattern)
    fnames.sort()
    if input_mask_regex is not None:
        fnames = filter_files(fnames,input_mask_regex)
    if len(fnames) == 0:
        raise ValueError(f"No files found for pattern {file_pattern}")
    for fname in fnames:
        print(fname)
        x = pd.read_csv(fname,parse_dates=['datetime'],sep=',')
        spot_check(x,fname)
        if len(x) == 0: 
            continue
        dss.append(x) 
    df = pd.concat(dss,axis = 0)
    df = df.reset_index(drop=True)
    
    return df


def spot_check(df,fname):
    if not("x2" in df.columns):
        raise ValueError(f"x2 missing in data frame for file {fname}")
