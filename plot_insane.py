import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 2)


info={0:1,58:2,59:4,60:5,61:10,62:20,66:3}#ensemble
#info={0:1,53:2,54:4,55:5,56:10,57:20}#sense
#info={0:0,63:np.log(5),64:np.log(50)}#initialize




attributes={"rank1":["acc","1"],"corr":["collapse","|corr|"],"trivial_features":["collapse","trivial_features"]}

mosaic=[gs[0,:],gs[1,0],gs[1,1]]
mosaic=[gs[:,0],gs[0,1],gs[1,1]]

datasets=["metal","pallet","market"]


baseline=[[0.595,0.34,63.3],[0.261,0.511,58.6]]#wait who cares
compress=[[0.806,0.268,40.6],[0.326,0.504,22.9],[0.478,0.288,30.0]]


def plotone(x,y,s,label,color):
    if np.max(np.abs(s))<0.0001:
        plt.plot(x,y,'o',markeredgewidth=1,label=label,color=color)
    else:
        plt.errorbar(x,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label=label,color=color)

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

plt.figure(figsize=(10,10))
for i,((attribute,keys),label) in enumerate(zip(attributes.items(),["Rank-1 Accuracy",r"$mAC$","Correlated Features"])):
    plt.subplot(mosaic[i])
    plt.xticks(list(info.values()))
    plt.xlabel("Amount of Submodels")
    if i==1:
        plt.xticks([1,2,3,4,5],[""]*5)
        plt.xlabel("")
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')

    if i:
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.yticks([])
        plt.gca().twinx()
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
    for j,dataset in enumerate(datasets):
        x,y,s=prepare_dic(dataset,info,*keys)
        plotone(x,y,s,label="Dataset "+("A" if dataset=="metal" else "B" if dataset=="pallet" else "C"),color=dataset_to_color(dataset))
        #plt.axhline(np.mean(y),color=dataset_to_color(dataset),linestyle="--",alpha=0.5)
        plt.axhline(compress[j][i],color=dataset_to_color(dataset),linestyle="--",alpha=0.5)
    plt.ylabel(label)
    if i==0:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(handles[0])
        labels.append(labels[0])
        handles=handles[1:]
        labels=labels[1:]
        handles.append(plt.axhline(-1,color="black",linestyle="--",alpha=0.5))
        labels.append("Compression")

        plt.legend(handles,labels,loc="best",frameon=True,framealpha=0.8)

    if i==0:
        yt=[0.0,0.25,0.5,0.75,1.0]
        yl=[r"$0\%$",r"$25\%$",r"$50\%$",r"$75\%$",r"$100\%$"]
        plt.yticks(yt,yl)
    if i==1:
        yt=[0.25,0.35,0.45,0.55]
        yl=[r"$25\%$",r"$35\%$",r"$45\%$",r"$55\%$"]
        plt.yticks(yt,yl)
    if i==0:
        plt.ylim(range_by_key(attribute))
    elif i==1:
        plt.ylim(0.25,0.6)
    else:
        plt.ylim(0,75)

    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')



plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








