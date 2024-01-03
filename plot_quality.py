import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys
import os
from basemodel import calc_accuracy
import json

from collapse import metrics as base_metrics


name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]


datasets=["metal","pallet","market"]


plt.figure(figsize=(10,5))
for dataset in datasets:
    with open(f"storage/ens_{dataset}.json","r") as f:
        data=json.load(f)
    
    
    def multiaccess(arr,*keys):
        for key in keys:
            try:
                arr=arr[key]
            except:
                print("key",key,"not found in",arr.keys())
                exit()
        return arr
    
    
    attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["|corr|"],"trivial_features":["trivial_features"]}
    
    
    toshow="rank1"
    sizes=data["x"]#list(range(25,maxima,125))#+[maxima]
    info=data["y"]#{size:metrics_by_size(size) for size in sizes}
    values=[multiaccess(info[str(size)],*attributes[toshow]) for size in sizes]
    
    x=np.array(sizes)
    y=np.array([v[0] for v in values])
    s=np.array([v[1] for v in values])
    if dataset=="market":x=x//4
    
    if dataset=="market":
        plt.plot(x//25,y,'o',label="Dataset "+("A" if dataset=="metal" else "B" if dataset=="pallet" else "C"),color=dataset_to_color(dataset))
    else:
        plt.errorbar(x//25,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label="Dataset "+("A" if dataset=="metal" else "B" if dataset=="pallet" else "C"),color=dataset_to_color(dataset))
    plt.axhline(y[-1],color=dataset_to_color(dataset),linestyle="--",alpha=0.5)
    
    

plt.xlabel("Amount of Submodels")
plt.ylabel("Rank-1 Accuracy")
plt.ylim(0,1)
plt.xlim(0,40)

yt=[0,0.25,0.5,0.75,1]
yl=[r"$0\%$",r"$25\%$",r"$50\%$",r"$75\%$",r"$100\%$"]
plt.yticks(yt,yl)

xt=[5,10,15,20,25,30,35,40]
plt.xticks(xt,xt)

handles,labels=plt.gca().get_legend_handles_labels()
order=[1,2,0]
handles=[handles[i] for i in order]
labels=[labels[i] for i in order]

plt.legend(handles,labels,frameon=True,framealpha=0.8)
    
plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")
    
    
    
plt.show()
    
    
    
    
    
    
    
                                                                                                                                                                                                                            
