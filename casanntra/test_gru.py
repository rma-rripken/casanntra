from read_data import read_data
from model_builder import *
from xvalid import xvalid_fit



def test_xvalid_gru():

    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","dcc","smscg"]
    output_names = ["x2","pct", "mal", "cse","anh","emm2","srv","rsl","oh4","trp","dsj","hll","bdl"]
    output_names = ["x2","mal","cse","emm2","tms","jer","rsl","bdl","oh4","srv"]
    output_names = ["x2","mal","nsl2","bdl","cse","emm2","tms","jer","sal","rsl","oh4"]
    builder = GRUBuilder(input_names=input_names,output_names=output_names,ndays=80)


    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)

    df_in, df_out = builder.xvalid_time_folds(df,target_fold_len='180d',split_in_out=True)   # adds a column called 'fold'

    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:,col] = df_out.loc[:,col]/100. 
        elif col in ["mal","cse","bdl","nsl2"]:
            df_out.loc[:,col] = df_out.loc[:,col]/10000.
        elif col in ["emm2","jer","tms"]:
            df_out.loc[:,col] = df_out.loc[:,col]/2000.
        else:
            df_out.loc[:,col] = df_out.loc[:,col]/1000.


    plot_locs = ["x2","cse","emm2","jer","bdl","sal","rsl"]
    xvalid_fit(df_in,df_out,builder,nepochs=50,plot_folds="all",plot_locs=plot_locs,out_prefix="gru")

if __name__ == "__main__":
   test_xvalid_gru()

