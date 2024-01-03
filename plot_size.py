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

plt.figure(figsize=(10,10))
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








