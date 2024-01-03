import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys

info={0:1,16:2,17:4,18:8}

baseline=np.array([100*val for val in info.values()])

attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["collapse","|corr|"],"not_trivial":["collapse","trivial_features"]}

shape=(2,2)

datasets=["metal","pallet"]

def plotone(x,y,s,label,color):
    plt.errorbar(x,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label=label,color=color)

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

plt.figure(figsize=(10,6))

tpl="not_trivial"
ds="metal"
col1="darkgreen"
col2="maroon"
x,y,s=prepare_dic(ds,info,*attributes[tpl])
plt.errorbar(x*100,y,s*10,fmt="--",label="Correlated Features (10x error)",color=col1)
#marker rund, weiß gefüllt, schwarzer rand
plt.errorbar(x*100,y,s*10,fmt="o",color=col1,markerfacecolor="white",markeredgecolor=col1,markeredgewidth=1)
plt.fill_between(x*100,0,y,alpha=0.2,color=col1)

plt.plot(x*100,100*x,label="Maximum Features",color="black")
handles, labels = plt.gca().get_legend_handles_labels()
plt.xlim(left=97,right=803)
plt.ylim(bottom=0)
plt.yticks([200,400,600,800],color=col1)
plt.xlabel("Representation Dimensionality")
plt.ylabel("Features",color=col1)

plt.gca().twinx()
tpl="corr"
x,y,s=prepare_dic(ds,info,*attributes[tpl])
plt.errorbar(x*100,y,s*10,fmt="--",label=r"$mAC$ (10x error)",color=col2)
plt.errorbar(x*100,y,s*10,color=col2,markerfacecolor="white",markeredgecolor=col2,markeredgewidth=1,fmt="o")
plt.ylabel(r"$mAC$",color=col2)
plt.ylim(0,1)
yt=[0.25,0.5,0.75,1]
yl=[r"$25\%$",r"$50\%$",r"$75\%$",r"$100\%$"]
plt.yticks(yt,yl,color=col2)

handles2, labels2 = plt.gca().get_legend_handles_labels()
plt.legend(handles+handles2, labels+labels2,loc="upper left")



plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")
plt.show()
exit()

for i,(attribute,keys) in enumerate(attributes.items()):
    plt.subplot(*shape,i+1)
    plt.xticks(list(info.values()))
    plt.xlabel(name)
    if i<2:
        plt.xticks([])
        plt.xlabel("")
    if i%2==1:
        plt.yticks([])
        plt.gca().twinx()
    for dataset in datasets:
        x,y,s=prepare_dic(dataset,info,*keys)
        if attribute=="not_trivial":
            y=baseline-y
        plotone(x,y,s,label=dataset,color=dataset_to_color(dataset))
    plt.ylabel(attribute)
    plt.ylim(range_by_key(attribute))
    if i==0:
        plt.legend()



plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








