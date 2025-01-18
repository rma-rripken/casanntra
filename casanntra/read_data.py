import pandas as pd
import os
import glob

data_repo = os.path.split(__file__)[0]+"/../data"

def read_data(file_pattern,data_dir = data_repo):
    dss = []
    file_pattern = os.path.join(data_dir,file_pattern)
    fnames = glob.glob(file_pattern)
    fnames.sort()
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

    
    return df

