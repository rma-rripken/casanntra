from casanntra.model_builder import *
import concurrent.futures
import matplotlib.pyplot as plt
from casanntra.single_or_list import *
import traceback
import pickle
import base64


def single_model_fit(
    builder,
    df_in,
    fit_in,
    fit_out,
    test_in,
    test_out,
    out_prefix,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
):
    input_layers = builder.input_layers()
    # Todo: train_model is the augmented model that includes a pass through of output
    #       ann is the main model for inference
    ann = builder.build_model(input_layers, df_in)
    # todo: this was train_model
    print("Fitting model in single_model_fit")
    history, ann = builder.fit_model(
        ann,
        fit_in,
        fit_out,
        test_in,
        test_out,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs,
    )

    # Todo
    print("Predicting data in single_model_fit")
    test_pred = ann.predict(test_in)
    # history_serialized = base64.b64encode(pickle.dumps(history.history)).decode()
    print("Prediction complete")
    del ann
    print(f"Return type {type(test_pred)}")
    return test_pred  # , history_serialized


def bulk_fit(
    builder,
    df_in,
    df_out,
    out_prefix,
    fit_in,
    fit_out,
    test_in,
    test_out,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
):
    """Uses the ingredients of xvalid_multi but does a single fit with all the data
    for situations like exporting the model where a single version of the model is needed.
    """

    # ✅ Ensure df_in and df_out are correctly formatted
    df_in = df_in.copy()
    df_in["ifold"] = 0

    if isinstance(df_out, list):
        df_out = [x.copy() for x in df_out]
        for df in df_out:
            df["ifold"] = 0

    # ✅ Apply antecedent preservation to aligned inputs
    # There will be only one set of inputs to the ANN (no source and target like outputs)
    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)
    outputs_trim = (
        [df.loc[inputs_lagged.index, builder.output_list()] for df in df_out]
        if isinstance(df_out, list)
        else df_out.loc[inputs_lagged.index, builder.output_list()]
    )

    fit_in = inputs_lagged
    fit_out = outputs_trim
    test_in = fit_in
    test_out = outputs_trim

    # ✅ Convert DataFrame inputs into structured dicts for multi-input models
    idx = pd.IndexSlice
    fit_in = builder.df_by_feature_and_time(fit_in).drop(
        ["datetime", "case", "fold"], level="var", axis=1
    )
    fit_in = {
        name: fit_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }
    test_in = builder.df_by_feature_and_time(test_in).drop(
        ["datetime", "case", "fold"], level="var", axis=1
    )
    test_in = {
        name: test_in.loc[:, idx[name, :]].droplevel("var", axis=1)
        for name in builder.input_names
    }

    input_layers = builder.input_layers()
    ann = builder.build_model(input_layers, df_in)

    history, ann = builder.fit_model(
        ann,
        fit_in,
        fit_out,
        test_in,
        test_out,
        init_train_rate=init_train_rate,
        init_epochs=init_epochs,
        main_train_rate=main_train_rate,
        main_epochs=main_epochs,
    )

    test_pred = ann.predict(test_in)

    return ann

    # xvalid_fit_multi(df_in,df_out,builder,plot_folds="all",plot_locs=plot_locs,
    #                 out_prefix=output_prefix, init_train_rate=init_train_rate,
    #                 init_epochs=init_epochs main_train_rate=None, main_epochs=-1, pool_size=pool_size)


def xvalid_fit_multi(
    df_in,
    df_out,
    builder,
    init_train_rate,
    init_epochs,
    main_train_rate,
    main_epochs,
    out_prefix,
    pool_size,
):
    """Splits up the input by fold, withholding each fold in turn, building and training the model,
    and then evaluating the withheld data."""

    num_outputs = builder.num_outputs()  # Get number of ANN outputs
    output_list = builder.output_list()

    # df_in will not be a list at this point
    # todo: it could contain entries for which one df_out has no outputs
    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)

    # df_in will not be a list at this point
    if isinstance(df_out, list):
        outputs_trim = [dfo.loc[inputs_lagged.index, :] for dfo in df_out]
    else:
        outputs_trim = df_out.loc[inputs_lagged.index, :]

    # create an empty data frame matching the output columns,
    # trimmed output rows plus datetime,case and fold
    # returns list if df_out is a list

    outputs_xvalid = allocate_receiving_df(outputs_trim, output_list)  # should

    futures = []
    foldmap = {}
    histories = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        for ifold in df_in.fold.unique():
            print(f"Scheduling fit for fold {ifold}")
            fit_in = inputs_lagged.loc[inputs_lagged.fold != ifold, :]
            test_in = inputs_lagged.loc[inputs_lagged.fold == ifold, :]

            if isinstance(df_out, list):
                fit_out = [df.loc[fit_in.index, output_list] for df in df_out]
                test_out = [df.loc[test_in.index, output_list] for df in df_out]
            else:
                fit_out = df_out.loc[fit_in.index, output_list]
                test_out = df_out.loc[test_in.index, output_list]

            if isinstance(fit_out, list):
                rep = fit_out[0] if isinstance(fit_out, list) else fit_out
                print(
                    f"ifold={ifold} # train input data rows = {fit_in.shape[0]} # train out rows = {rep.shape[0]}"
                )
                print(
                    f"ifold={ifold} # test input data rows = {test_in.shape[0]} # test out rows = {rep.shape[0]}"
                )

            # ✅ Convert DataFrame inputs into structured dicts
            idx = pd.IndexSlice
            # These libraries separate the inputs into individual dataframes matching the input names
            fit_in = builder.df_by_feature_and_time(fit_in).drop(
                ["datetime", "case", "fold"], level="var", axis=1
            )
            fit_in = {
                name: fit_in.loc[:, idx[name, :]].droplevel("var", axis=1)
                for name in builder.input_names
            }

            test_in = builder.df_by_feature_and_time(test_in).drop(
                ["datetime", "case", "fold"], level="var", axis=1
            )
            test_in = {
                name: test_in.loc[:, idx[name, :]].droplevel("var", axis=1)
                for name in builder.input_names
            }

            future = executor.submit(
                single_model_fit,
                builder,
                df_in,
                fit_in,
                fit_out,
                test_in,
                test_out,
                out_prefix=out_prefix,
                init_epochs=init_epochs,
                init_train_rate=init_train_rate,
                main_epochs=main_epochs,
                main_train_rate=main_train_rate,
            )
            futures.append(future)
            foldmap[future] = ifold

    for future in concurrent.futures.as_completed(futures):
        try:
            ifold = foldmap[future]
            test_pred = future.result()
            # history = pickle.loads(base64.b64decode(history_encoded))
            test_in = inputs_lagged.loc[inputs_lagged.fold == ifold, :]
            if isinstance(outputs_xvalid, list):
                for i in range(num_outputs):
                    print(
                        "\nUpdating masxter xvalidation data structure (multiple output version)"
                    )
                    outputs_xvalid[i].loc[test_in.index, output_list] = test_pred[i]
            else:
                print(
                    "\nUpdating master xvalidation data structure (single output version)"
                )
                outputs_xvalid.loc[test_in.index, output_list] = test_pred
                print("Done")
        except Exception as err:
            print(f"Exception in (probably) in fold: {ifold}")
            traceback.print_tb(err.__traceback__)
            raise err

    full_col_list = ["datetime", "case", "fold"] + output_list
    print("Writing master xvalidation data structure to file")
    if isinstance(outputs_xvalid, list):
        print("Multiple structures")
        for i in range(num_outputs):
            outxfile = f"{out_prefix}_xvalid_{i}.csv"
            outputs_xvalid[i][output_list] = outputs_xvalid[i][output_list].astype(
                float
            )
            print(f"writing to file {i} {outxfile}")
            print(outputs_xvalid[i])
            outputs_xvalid[i].to_csv(
                outxfile,
                float_format="%.3f",
                date_format="%Y-%m-%dT%H:%M",
                header=True,
                index=True,
            )
            print("and now ")
            print(outputs_xvalid[i])
    else:
        print("Single Structure")
        outputs_xvalid[output_list] = outputs_xvalid[output_list].astype(float)
        outputs_xvalid.to_csv(
            f"{out_prefix}_xvalid.csv",
            float_format="%.3f",
            date_format="%Y-%m-%dT%H:%M",
            header=True,
            index=True,
        )
    print("Done writing\n\n")
    return outputs_xvalid, histories


@single_or_list("df_out")
def allocate_receiving_df(df_out, column_list):
    # ✅ Pre-allocate multiple datastructures for holding results based on num_outputs
    output_cols = ["datetime", "case", "fold"] + column_list
    outputs_xvalid = pd.DataFrame(columns=output_cols, index=df_out.index)
    # ✅ Preserve datetime, case, and fold info
    outputs_xvalid.loc[:, ["datetime", "case", "fold"]] = df_out.loc[
        :, ["datetime", "case", "fold"]
    ]
    return outputs_xvalid
