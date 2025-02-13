import pandas as pd
import os
import re
import glob

data_repo = os.path.split(__file__)[0] + "/../data"


def read_data(file_pattern, input_mask_regex, data_dir=data_repo):
    """
    Reads and concatenates multiple CSV files matching a given pattern.

    This function loads time-series data from CSV files, applies optional filtering,
    and performs basic validation.

    Parameters
    ----------
    file_pattern : str
        A filename pattern (e.g., `"dataset_*.csv"`) used to match files.
    input_mask_regex : str or list of str or None
        Regular expression(s) to filter out specific files.
        If `None`, all matching files are loaded.
    data_dir : str, optional
        Base directory where files are located (default is `data_repo`).

    Returns
    -------
    pandas.DataFrame
        A concatenated dataframe containing all matching data.

    Raises
    ------
    ValueError
        If no files match the given pattern.
        If any loaded file is missing the required `"x2"` column.

    Examples
    --------
    >>> df = read_data("dataset_*.csv", input_mask_regex="ignore_this.*")
    >>> df.head()

    Notes
    -----
    - Uses `glob.glob()` to match files in `data_dir`.
    - Calls `filter_files()` to exclude files based on `input_mask_regex`.
    - Loads CSV files with `pandas.read_csv()` and parses `"datetime"` columns.
    - Calls `spot_check()` to validate required columns.
    - Resets the index before returning the concatenated dataframe.
    """
    dss = []
    file_pattern = os.path.join(data_dir, file_pattern)
    fnames = glob.glob(file_pattern)
    fnames.sort()
    if input_mask_regex is not None:
        fnames = filter_files(fnames, input_mask_regex)
    if len(fnames) == 0:
        raise ValueError(f"No files found for pattern {file_pattern}")
    for fname in fnames:
        print(fname)
        x = pd.read_csv(fname, parse_dates=["datetime"], sep=",")
        spot_check(x, fname)
        if len(x) == 0:
            continue
        dss.append(x)
    df = pd.concat(dss, axis=0)
    df = df.reset_index(drop=True)

    return df



def filter_files(fnames, masks, verbose=False):
    """
    Filters a list of filenames based on specified exclusion masks.

    This function removes filenames that match any of the provided regular expressions.

    Parameters
    ----------
    fnames : list of str
        List of filenames (including paths) to be filtered.
    masks : str or list of str or None
        Regular expressions defining patterns for exclusion.
        If `None`, no filtering is applied.
    verbose : bool, optional
        If `True`, prints debug messages showing which files are excluded.

    Returns
    -------
    list of str
        Filtered list of filenames that do not match any exclusion mask.

    Examples
    --------
    >>> fnames = ["data_1.csv", "data_2.csv", "exclude_this.csv"]
    >>> masks = "exclude.*\\.csv"
    >>> filter_files(fnames, masks)
    ['data_1.csv', 'data_2.csv']

    Notes
    -----
    - If `masks` is a string, it is converted to a list.
    - Regular expressions are applied to the base filename, not the full path.
    - Matching is case-sensitive.
    """    
    if masks is None or masks == "None":
        return fnames
    if isinstance(masks, str):
        masks = [masks]
    mask_res = [re.compile(x) for x in masks]

    def exclude(mask_res, f):
        masked = any([mr.match(os.path.split(f)[1]) for mr in mask_res])
        if verbose:
            print(f"file {f}. Masking = {masked}")
        return masked

    return [f for f in fnames if not exclude(mask_res, f)]



def compute_scenario_differences(df, base_col, suisun_col):
    """
    Computes the difference between two scenarios for each case.

    This function calculates the difference `(suisun_col - base_col)`
    for each case in the dataset. If one of the scenarios is missing,
    it falls back to using absolute values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing multiple cases.
    base_col : str
        Column name representing the base scenario.
    suisun_col : str
        Column name representing the Suisun scenario.

    Returns
    -------
    pandas.DataFrame
        Modified dataframe with an additional column `'scenario_difference'`.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {"case": [1, 1, 2, 2], "base": [10, 20, 30, 40], "suisun": [15, 25, 35, 45]}
    >>> df = pd.DataFrame(data)
    >>> compute_scenario_differences(df, "base", "suisun")
       case  base  suisun  scenario_difference
    0     1    10      15                   5
    1     1    20      25                   5
    2     2    30      35                   5
    3     2    40      45                   5

    Notes
    -----
    - If both `base_col` and `suisun_col` exist, computes their difference.
    - If only `base_col` exists, assigns its values to `'scenario_difference'`.
    - If only `suisun_col` exists, assigns its values to `'scenario_difference'`.
    - This ensures all cases have a valid comparison value.
    """
    df = df.copy()
    df["scenario_difference"] = None  # Initialize new column

    available_cases = df.groupby("case")

    for case, group in available_cases:
        base_exists = base_col in group.columns
        suisun_exists = suisun_col in group.columns

        if base_exists and suisun_exists:
            df.loc[group.index, "scenario_difference"] = (
                group[suisun_col] - group[base_col]
            )
        elif base_exists:
            df.loc[group.index, "scenario_difference"] = group[
                base_col
            ]  # Fall back to absolute MAE
        elif suisun_exists:
            df.loc[group.index, "scenario_difference"] = group[
                suisun_col
            ]  # Fall back to absolute MAE

    return df



def spot_check(df, fname):
    if not ("x2" in df.columns):
        raise ValueError(f"x2 missing in data frame for file {fname}")
