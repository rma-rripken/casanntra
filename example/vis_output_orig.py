import matplotlib.pyplot as plt
import pandas as pd
import sys

output_prefixs = {}
output_prefix0 = "dsm2.schism_base_gru2"
output_prefix1 = "dsm2_base_gru2"
output_prefixs["dsm2.schism"] = output_prefix0
output_prefixs["dsm2"] = output_prefix1


station = "x2"
if len(sys.argv) > 1: station = sys.argv[1]
if len(sys.argv) > 2: 
   model = sys.argv[2]
   output_prefix = output_prefixs[model]

#model_data=pd.read_csv("output/schism_mlp1m.mae_xvalid_ref_out.csv",index_col=0, parse_dates=['datetime'],header=0)
#ann_data=pd.read_csv("output/schism_mlp1m.mae_xvalid.csv",index_col=0, parse_dates=['datetime'],header=0)
model_data=pd.read_csv(f"{output_prefix}_xvalid_ref_out_scaled.csv",index_col=0, parse_dates=['datetime'],header=0)
ann_data=pd.read_csv(f"{output_prefix}_xvalid_.csv",index_col=0, parse_dates=['datetime'],header=0)



full_names = {
  "cse": "Collinsville",
  "rsl":"Rock slough",
  "oh4": "Old@HW4",
  "frk": "Franks Tract",
  "bac": "Old R. at Bacon",
  "x2": "X2",
  "emm2": "Emmaton",
  "jer" : "Jersey Point",
  "bdl" : "Beldon's Landing",
  "mal" : "Mallard",
  "hll" : "Holland Cut",
  "sal" : "San Andreas Landing",
  "bdl" : "Beldon's Landing",
  "god" : "Godfather Sl",
  "gzl" : "Grizzly Bay",
  "vol" : "Suisun Sl at Volanti"
}


title = full_names[station]


ncase = 7

fig,axes = plt.subplots(ncase,sharex = False,constrained_layout=True,figsize=(8,9))
nax = len(axes)

for i in range(ncase):
    print(i)
    icase = i + 1 +i*10 + 11 
    icase = i+1
    sub_mod = model_data.loc[model_data.case==icase,:]
    sub_ann = ann_data.loc[ann_data.case==icase,:]
    ax = axes[i]
    if i == 0: ax.set_title(title)
    ax.plot(sub_mod.datetime,sub_mod[station])
    ax.plot(sub_ann.datetime,sub_ann[station])
    ax.set_ylabel("Norm EC")
    if i == 0:
       ax.set_title(f"Station={station}, Case = {icase}")
    else:
        ax.set_title(f"Case = {icase}")

plt.tight_layout()
plt.show()