import pandas as pd
import os
import re
import glob

data_repo = os.path.split(__file__)[0]+"/../data"

def filter_files(fnames,masks):
    if masks is None or masks == "None": 
        return fnames
    mask_res=[re.compile(x) for x in masks]
    def exclude(mask_res,f):
        print(os.path.split(f)[1])
        return any([mr.match(os.path.split(f)[1]) for mr in mask_res])
    return [f for f in fnames if not exclude(mask_res, f)]


def read_data(file_pattern,input_mask_regex,data_dir = data_repo):
    """ Read data
        file_pattern is a pattern for matching
        input_mask is a list of masks to use to filter out excluded cases
        
    """
    dss = []
    file_pattern = os.path.join(data_dir,file_pattern)
    fnames = glob.glob(file_pattern)
    fnames.sort()
    fnames = filter_files(fnames,input_mask_regex)
    if len(fnames) == 0:
        raise ValueError(f"No files found for pattern {file_pattern}")
    for fname in fnames:
        print(fname)
        x = pd.read_csv(fname,parse_dates=['datetime'],sep=',')
        if len(x) == 0: 
            continue
        dss.append(x) 
    df = pd.concat(dss,axis = 0)
    df = df.reset_index(drop=True)
    if "dsm2" in file_pattern:
        for i in range(101,105):
            df.loc[df.case == i,"sf_tidal_filter"] = df.loc[df.case == i,"mrz_tidal_filter"]
            df.loc[df.case == i,"sf_tidal_energy"] = df.loc[df.case == 23,"sf_tidal_energy"].values   
        for i in range(105,108):
            df.loc[df.case == i,"sf_tidal_filter"] = df.loc[df.case == i,"mrz_tidal_filter"]
            df.loc[df.case == i,"sf_tidal_energy"] = df.loc[df.case == 45,"sf_tidal_energy"].values
    else:
        df.loc[:,"delta_cu"]*=-1.
        df.loc[:,"exports"]*=1.
    
    return df

