from casanntra.staged_learning import process_config

def main():
    configfile = "transfer_config.yml"
    process_config(configfile, ["dsm2_base", "dsm2.schism", "base.suisun"])
    #process_config(configfile, ["base.suisun"])
    #process_config(configfile, ["dsm2_base"])

if __name__ == "__main__":
    main()