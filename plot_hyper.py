import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys


#first lr
info1={29:0.0001,30:0.01,0:0.001,41:0.002,42:0.005,43:0.0002,44:0.0005}
info1={29:0.0001,0:0.001,41:0.002,43:0.0002,44:0.0005}
#then act
info2={0:"relu",31:"LeakyReLU",33:"elu"}#,46:"sigmoid"}

attributes={"rank1":["acc","1"],"rank10":["acc","10"],"corr":["collapse","|corr|"],"trivial_features":["collapse","trivial_features"]}

shape=(2,2)

datasets=["metal","pallet"]

def plotone(x,y,s,label,color):
    plt.errorbar(x,y,s,fmt='o',capsize=5,elinewidth=1,markeredgewidth=1,label=label,color=color)

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]

plt.figure(figsize=(10,11))
#3 wide, 2 high. (rank1, corr, trivial_features) x (info, info2)

def gen_label(ds):
    if ds=="metal":
        return "A"
    elif ds=="pallet":
        return "B"
    elif ds=="market":
        return "C"
    else:
        return ds

def translate(x):
    if type(x) is str or type(x) is np.str_:
        if x.lower()=="leakyrelu":
            return "LeakyReLU"
        elif x.lower()=="relu":
            return "ReLU"
        elif x.lower()=="elu":
            return "ELU"
    else:
        return np.array([translate(xx) for xx in x])

def weighted_mean(y,s):
    return np.sum(y/s**2)/np.sum(1/s**2)
def chisquare(y,s,mn):
    return np.sum(((y-mn)/s)**2)/(len(y)-1)

for j,(info,label) in enumerate(zip([info1,info2],["Learning Rate","Activation Function","Learning Rate"])):
    for i,(attribute,alabel) in enumerate(zip(["rank1","corr","trivial_features"],["Rank-1 Accuracy",r"$mAC$","Correlated Features"])):
        plt.subplot(3,2,2*i+j+1)
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        if i==2:
            plt.gca().set_xlabel(label)
        #else:
        #    plt.xticks([])
        if j==1:
            plt.yticks([])
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.gca().twinx()
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')

        for dataset in datasets:
            x,y,s=prepare_dic(dataset,info,*attributes[attribute])
            if j==1:
                x=translate(x)
            plotone(x,y,s,label="Dataset "+gen_label(dataset),color=dataset_to_color(dataset))
            if i<2:
                plt.axhline(y=weighted_mean(y,s),color=dataset_to_color(dataset),linestyle="--",alpha=0.5)
                chi=chisquare(y,s,weighted_mean(y,s))
                print(f"{dataset} {attribute} {label} {chi:.2f}")
        
        plt.ylim(range_by_key(attribute))



        if j==0:
            plt.xscale("log")

        if i==2 and j==1 and False:
            plt.legend(frameon=True,loc="lower right",framealpha=0.8)
        if i==1 and j==0 and True:
            plt.legend(frameon=True,loc="upper center",framealpha=0.8)
            #plt.xticks(rotation=45)

        if i==2:
            plt.gca().set_xlabel(label)
        elif j==0:
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.xticks([])
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
        else:
            #plt.gca().xaxis.set_
            plt.xticks([0,1,2],["","",""])

        if j==0 or j==1:
            plt.ylabel(alabel)

        if i<2:
            yt=[0,0.25,0.5,0.75,1]
            yl=[r"${:.0f}\%$".format(y*100) for y in yt]
            plt.yticks(yt,yl)
        else:
            yt=[25,50,75,100]
            yl=[r"${:.0f}$".format(y) for y in yt]
            plt.yticks(yt,yl)

        if j==0 and False:
            xt=[1e-4,3e-4,1e-3,3e-3]
            xl=[r"${:.0e}$".format(x) for x in xt]
            plt.xticks(xt,xl)



        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')

plt.tight_layout()

plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")



plt.show()








