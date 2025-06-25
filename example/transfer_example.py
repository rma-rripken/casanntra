from casanntra.staged_learning import process_config


def main():
    """Ultimately we may launch this using argparse, but it is hard wired right now.
    The first argument, configfile, gives the configuration steps for DSM2 base, DSM2->SCHISM
    and SCHISM,base -> SCHISM,suisun.

    My visualization of the cross validation output is vis_output.py.
    It is in much more rudimentary state.
    """
    configfile = "transfer_config_schism_v4.yml"

    # This selects the steps you want to run
    process_config(configfile, ["dsm2_base", "dsm2.schism", "base.suisun"])
    # process_config(configfile, ["dsm2.schism", "base.suisun"])
    # process_config(configfile, ["dsm2.schism"])
    # process_config(configfile, ["base.suisun"])
    # process_config(configfile, ["dsm2_base"])


if __name__ == "__main__":
    main()