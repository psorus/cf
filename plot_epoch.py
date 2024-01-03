import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key, raw_access, dataset_to_color
import sys

info={0:1,34:2,35:4,36:6,38:3,39:5,40:7}
info=0

attributes={"corr":["timecollapse","|corr|"],"trivial_features":["timecollapse","trivial_features"]}
shape=(1,2)

datasets=["metal","pallet"]

def adaptive_mean(lis):
    counts=[]
    values=[]
    for slis in lis:
        for i,val in enumerate(slis):
            if i>=len(counts):
                counts.append(0)
                values.append(0)
            counts[i]+=1
            values[i]+=val
    return [val/count for val,count in zip(values,counts)]


def plotdic(dic,label):
    x=list(dic.keys())
    x.sort(key=int)
    y=[dic[k] for k in x]
    x=[int(k) for k in x]
    plt.plot(x,y,'o-',alpha=0.1,color=dataset_to_color(label))
    return y


def plotdics(dics,label):
    mn=adaptive_mean([plotdic(dic,label) for dic in dics])
    plt.plot(np.arange(len(mn)),mn,'-',alpha=1,color=dataset_to_color(label))
    plt.plot(np.arange(len(mn)),mn,'o',alpha=1,color=dataset_to_color(label),label="Dataset "+gen_label(label))

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

attr_to_nam={"corr":r"$mAC$","trivial_features":"Correlated Features"}

def gen_label(ds):
    if ds=="metal":
        return "A"
    elif ds=="pallet":
        return "B"
    elif ds=="market":
        return "C"
    else:
        return ds

plt.figure(figsize=(10,5))
for i,(attribute,keys) in enumerate(attributes.items()):
    plt.subplot(*shape,i+1)
    plt.xlabel("Number of Epochs")
    if i%2==1:
        plt.yticks([])
        plt.gca().twinx()
    for dataset in datasets:
        dics=raw_access(dataset,info,*keys)
        plotdics(dics,label=dataset)
    plt.ylabel(attr_to_nam[attribute])
    plt.ylim(range_by_key(attribute))
    if i==0:
        plt.legend()
    if i==0:
        yt=[0,0.25,0.5,0.75,1]
        yl=[r"$%d\%%$"%(y*100) for y in yt]
        plt.yticks(yt,yl)
    if i==1:
        yt=[0,25,50,75,100]
        plt.yticks(yt,yt)
    xt=[0,10,20,30,40]
    plt.xticks(xt,xt)




plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








