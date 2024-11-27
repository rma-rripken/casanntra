
import matplotlib.pyplot as plt
import pandas as pd

model_data=pd.read_csv("output/schism_mlp1m.mae_xvalid_ref_out.csv",index_col=0, parse_dates=['datetime'],header=0)
ann_data=pd.read_csv("output/schism_mlp1m.mae_xvalid.csv",index_col=0, parse_dates=['datetime'],header=0)


station = "cse"
title = "Collinsville"


station = "bdl"
title = "Beldon's Landing"
ncase = 7

fig,axes = plt.subplots(ncase,sharex = False,constrained_layout=True,figsize=(8,9))
nax = len(axes)

for i in range(ncase):
    print(i)
    icase = i + 1
    sub_mod = model_data.loc[model_data.case==icase,:]
    sub_ann = ann_data.loc[ann_data.case==icase,:]
    ax = axes[i]
    if i == 0: ax.set_title(title)
    ax.plot(sub_mod.datetime,sub_mod[station])
    ax.plot(sub_ann.datetime,sub_ann[station])
    ax.set_ylabel("Norm EC")

plt.show()

