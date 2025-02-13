from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit
from test_gru1 import GRUBuilder1
from keras.models import load_model


def example_gru1():

    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)
    input_names = [
        "sac_flow",
        "exports",
        "sjr_flow",
        "cu_flow",
        "sf_tidal_energy",
        "sf_tidal_filter",
        "dcc",
        "smscg",
    ]
    output_names = [
        "x2",
        "mal",
        "nsl2",
        "bdl",
        "cse",
        "emm2",
        "tms",
        "jer",
        "sal",
        "bac",
        "oh4",
    ]
    builder = GRUBuilder1(input_names=input_names, output_names=output_names, ndays=80)
    df_in = df[["datetime", "case"] + input_names]
    df_out = df[output_names]

    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:, col] = df_out.loc[:, col] / 100.0
        elif col in ["mal", "cse", "bdl", "nsl2"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 10000.0
        elif col in ["emm2", "jer", "tms"]:
            df_out.loc[:, col] = df_out.loc[:, col] / 2000.0
        else:
            df_out.loc[:, col] = df_out.loc[:, col] / 1000.0

    input_layers = builder.input_layers()
    model = builder.build_model(
        input_layers, df_in
    )  # todo: check that it is OK for datetime and cases to be in here
    df_lagged = builder.calc_antecedent_preserve_cases(df_in, ndays=80, reverse=False)
    timestamps_case = df_lagged[["datetime", "case"]]
    df_lag_in = df_lagged.drop(["datetime", "case"], axis=1)
    df_out_aligned = df_out.reindex(df_lag_in.index)

    idx = pd.IndexSlice
    fit_in = builder.df_by_feature_and_time(df_lag_in)
    fit_in = {
        name: fit_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }

    test_in = builder.df_by_feature_and_time(df_lag_in)
    test_in = {
        name: test_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }

    #
    fname = "full_data_fit.keras"
    # model = load_model(fname)
    history, model = builder.fit_model(
        model, fit_in, df_out_aligned, test_in, df_out_aligned, nepochs=130
    )

    model.save(fname)
    # copy to borrow size
    df_out_dated = df_out_aligned.join(timestamps_case)
    pred_out = df_out_dated.copy()
    print("shapes")
    print(df_out_dated.shape)
    print(pred_out.shape)
    print(pred_out.columns)
    predicted = model.predict(test_in)
    print(predicted.shape)
    pred_out.loc[:, builder.output_names] = model.predict(test_in)

    cases = df_out_dated.case.unique()
    for icase in df_out_dated.case.unique():
        print(f"Processing case {icase}")
        dfsub = df_out_dated.loc[df_out_dated.case == icase, :]
        predsub = pred_out.loc[pred_out.case == icase, :]
        plt.plot(dfsub.datetime, dfsub.cse)
        plt.plot(predsub.datetime, predsub.cse)
        plt.show()


if __name__ == "__main__":
    example_gru1()
