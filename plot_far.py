import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys

info={0:1,20:2,21:4,22:8,48:0.25,49:0.5,50:0.75,51:1.5,52:2.5}




attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["collapse","|corr|"],"trivial_features":["collapse","trivial_features"]}

shape=(2,2)

datasets=["metal","pallet"]

def plotone(x,y,s,label,color):
    plt.errorbar(x,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label=label,color=color)

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

plt.figure(figsize=(10,5))

dataset="metal"
#dataset="pallet"
#dataset="market"
att1,att2="rank1","corr"
col1,col2="darkblue","darkgreen"
#att2="trivial_features"

tobeat=0.806,0.012


x,y,s=prepare_dic(dataset,info,*attributes[att1])
plotone(x,y,s,color=col1,label=None)
plt.ylabel("Rank-1 Accuracy",color=col1)
plt.yticks(color=col1)
plt.xlabel("Node-count Multiplier")

plt.axhline(tobeat[0],color="purple",label="Accuracy (3x reduction)")
bd=np.where(x==1)[0][0]
#plt.axvline(x[bd],color="black",linestyle="--",label="Baseline")
#plt.axhline(y[bd],color="black",linestyle="--")

plt.annotate("Accuracy (3x reduction)",xy=(x[bd]+0.55,tobeat[0]),xytext=(x[bd]+0.05,tobeat[0]-0.08),arrowprops=dict(arrowstyle="->",color="black"))
plt.annotate("Baseline",xy=(x[bd]-0.03,y[bd]+0.01),xytext=(x[bd]-0.45,y[bd]+0.08),arrowprops=dict(arrowstyle="->",color="black"))




plt.gca().twinx()
x,y,s=prepare_dic(dataset,info,*attributes[att2])
plotone(x,y,s,color=col2,label=None)
plt.ylabel(r"$mAC$",color=col2)
plt.yticks(color=col2)





plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








