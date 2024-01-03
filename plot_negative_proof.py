import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys
from matplotlib import patches as mpat


info={0:1,34:2,35:4,36:6,38:3,39:5,40:7}


attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["collapse","|corr|"],"trivial_features":["collapse","trivial_features"]}

shape=(2,2)

datasets=["metal","pallet"]
datasets=["metal"]

def plotone(x,y,s,label,color):
    plt.errorbar(x,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label=label,color=color)

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

X="rank1"
Y="corr"
Y="trivial_features"

linrang=[0.5,0.9]
lin=np.linspace(*linrang,100)

plt.figure(figsize=(10,6))
for dataset in datasets:
    _,xx,sx=prepare_dic(dataset,info,*attributes[X])
    _,yy,sy=prepare_dic(dataset,info,*attributes[Y])
    #plt.errorbar(xx,yy,sy,sx,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label="Different Compressions",color=dataset_to_color(dataset))

    plt.plot(xx,yy,'o',markeredgewidth=1,label="Different Compressions",color=dataset_to_color(dataset))

    for xxx,yyy,sxx,syy in zip(xx,yy,sx,sy):
        plt.gca().add_patch(mpat.Ellipse((xxx,yyy),2*sxx,2*syy,fill=True,facecolor=dataset_to_color(dataset),edgecolor=dataset_to_color(dataset),alpha=0.3))

    #linear fit
    z=np.polyfit(xx,yy,1)
    corr=np.corrcoef(xx,yy)[0,1]
    print("Correlation",corr)
    p=np.poly1d(z)
    xlim=plt.gca().get_xlim()
    ylim=plt.gca().get_ylim()
    
    plt.plot(lin,p(lin),color=dataset_to_color(dataset),linestyle="--",alpha=0.7,label="Linear Fit")
    plt.xlim(*xlim)
    plt.ylim(*ylim)

xt=[0.55,0.6,0.65,0.7,0.75,0.8]
xl=[r"$55\%$",r"$60\%$",r"$65\%$",r"$70\%$",r"$75\%$",r"$80\%$"]
plt.xticks(xt,xl)


plt.ylabel("Correlated Features")
plt.xlabel("Rank-1 Accuracy")
#handles, labels = plt.gca().get_legend_handles_labels()
#plt.legend(handles[::-1],labels[::-1],frameon=True,loc="upper right",framealpha=0.8)
handles, labels = plt.gca().get_legend_handles_labels()
handles=handles[::-1]
labels=labels[::-1]
handles.append(mpat.Ellipse((0,0),1,1,fill=True,color=dataset_to_color(dataset),alpha=0.3))
labels.append("Area of Uncertainty")
plt.legend(handles,labels,frameon=True,loc="upper right",framealpha=0.8)
#plt.legend(handles,labels,frameon=True,loc="lower left",framealpha=0.8)



plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








