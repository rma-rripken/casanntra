
from casanntra.read_data import read_data
import matplotlib.pyplot as plt

fpattern = "dsm2_base_*.csv"
df = read_data(fpattern)

fig,((ax0,ax0a),(ax1,ax1a),(ax2,ax2a)) = plt.subplots(3,2)
ax0.hist(df.jer,bins=15,range=(0,4000))
ax0a.ecdf(df.jer, label="CDF")
ax0a.grid()
ax0.set_title("JER")

ax1.hist(df.emm2,bins=15,range=(0,6000))
ax1a.ecdf(df.emm2, label="CDF")
ax1a.grid()
ax1.set_title("EMM")

ax2.hist(df.bac,bins=15,range=(0,3000.))
ax2a.ecdf(df.bac, label="CDF")
ax2a.grid()
ax2.set_title("BAC")

plt.show()
