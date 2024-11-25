from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit
from test_gru1 import GRUBuilder1


def example_gru1():

    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)
    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","sf_tidal_filter","dcc","smscg"]
    output_names = ["x2","mal","nsl2","bdl","cse","emm2","tms","jer","sal","bac","oh4"]
    builder = GRUBuilder1(input_names=input_names,output_names=output_names,ndays=80)
    df_in = df[["datetime","case"]+input_names]   
    df_out = df[output_names]

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
    

    
    input_layers = builder.input_layers()
    model = builder.build_model(input_layers,df_in)       # todo: check that it is OK for datetime and cases to be in here
    df_lag_in = builder.calc_antecedent_preserve_cases(df_in, ndays=80, reverse=False)
    timestamps = df_lag_in.datetime
    df_lag_in = df_lag_in.drop(["datetime","case"],axis=1)
    df_out_aligned = df_out.reindex(df_lag_in.index)          

    idx = pd.IndexSlice
    fit_in = builder.df_by_feature_and_time(df_lag_in)
    fit_in =  {name: fit_in.loc[:,idx[name,:]].droplevel("var",axis=1) for name in builder.input_names}


    test_in = builder.df_by_feature_and_time(df_lag_in)
    test_in =  {name: test_in.loc[:,idx[name,:]].droplevel("var",axis=1) for name in builder.input_names}    

    # This will produce bogus validation numbers, so you need to know # of epochs
    model = builder.fit_model(model, fit_in, df_out_aligned, test_in, df_out_aligned)


if __name__ == "__main__":
   example_gru1()

