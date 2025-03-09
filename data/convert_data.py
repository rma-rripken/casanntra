import glob
import pandas as pd
import numpy as np
import os
import argparse

def check_and_fix_file(file_path, strict=False):
    if "sf_tide" in file_path: return
    # Read the CSV file; assume the first column 'datetime' is parseable as dates.
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    modified = False  # Flag to indicate if changes were made
    
    # --- Check cu_flow for July 1 ---
    # Filter rows where the date is July 1
    july1_mask = (df['datetime'].dt.month == 7) & (df['datetime'].dt.day == 1)
    july1_data = df.loc[july1_mask, 'cu_flow']
    
    if not july1_data.empty:
        # Compute the 2nd percentile value of cu_flow for July 1
        cu_flow_2perc = np.percentile(july1_data, 2)
        if cu_flow_2perc <= 0:
            message = (f"In file '{os.path.basename(file_path)}': 2nd percentile of "
                       f"cu_flow on July 1 is {cu_flow_2perc:.2f} (<=0), so reversing cu_flow sign.")
            if strict:
                raise ValueError(message)
            else:
                df['cu_flow'] = df['cu_flow'] * -1
                print(message)
                modified = True
    else:
        print(f"In file '{os.path.basename(file_path)}': No July 1 data found for cu_flow check.")
    
    # --- Check exports column ---
    exports_min = df['exports'].min()
    if exports_min <= 0:
        message = (f"In file '{os.path.basename(file_path)}': Minimum exports value is {exports_min:.2f} "
                   f"(<=0), so reversing exports sign.")
        if strict:
            raise ValueError(message)
        else:
            df['exports'] = df['exports'] * -1
            print(message)
            modified = True

    # If modifications were made, overwrite the file (or you could choose to write to a new file)
    if modified:
        df.to_csv(file_path, index=False,float_format="%.2f",date_format="%Y-%m-%d")
        print(f"File '{os.path.basename(file_path)}' has been updated.\n")
    else:
        print(f"File '{os.path.basename(file_path)}' passed all checks.\n")

def main(directory, strict=False):
    # Glob all CSV files in the given directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    for file in csv_files:
        try:
            check_and_fix_file(file, strict)
        except Exception as e:
            print(f"Error processing file '{os.path.basename(file)}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check and fix sign issues in CSV files for 'cu_flow' (July 1) and 'exports' columns."
    )
    parser.add_argument("directory", help="Directory containing CSV files")
    parser.add_argument(
        "--strict", 
        action="store_true", 
        help="Raise an exception if any check fails instead of auto-correcting."
    )
    args = parser.parse_args()
    main(args.directory, args.strict)
