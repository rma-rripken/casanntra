from casanntra.fit_from_config import process_config

def main():
    configfile = "transfer_config.yml"
    process_config(configfile, ["dsm2_base", "dsm2.schism", "base.suisun"])

if __name__ == "__main__":
    main()