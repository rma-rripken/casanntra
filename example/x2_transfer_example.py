
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


import multiprocessing as mp
from casanntra.staged_learning import process_config



def main():
    configfile = "x2_transfer_config.yml"

    process_config(configfile, ["x2_dsm2_base", "x2_dsm2.schism", "x2_base.suisun", "x2_dsm2.rma", "x2_rma_base.suisun", "x2_rma_base.cache", "x2_rma_base.ft"])
    # "x2_dsm2.rma", "x2_rma_base.ft", "x2_rma_base.cache"])
    #process_config(configfile, ["x2_dsm2.schism"])
    #process_config(configfile, ["x2_base.suisun"])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

