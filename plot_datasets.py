import numpy as np
from plt import *
import sys
import os

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]


dic={"A":"smalldata/metal/fold0.npz","B":"smalldata/pallet/light_on_0.npz","C":"smalldata/market/market.npz"}

def load_one(fn):
    f=np.load(fn)
    x=f["x"]
    return x[1]

dic={k:load_one(v) for k,v in dic.items()}
#dic["B"]=dic["B"].transpose(1,0,2)


plt.figure(figsize=(10,3))
count=len(list(dic.values()))
heis=[v.shape[0]/v.shape[1] for v in dic.values()]
print(heis)
heis=np.array(heis)
heis=heis/heis.max()
heis[1]=0.3
minpos=0.1
for i,((k,v),hei) in enumerate(zip(dic.items(),heis)):
    #plt.subplot(1,count,i+1)
    #plt.title(k)
    wid=1/hei
    plt.imshow(v,extent=(minpos,minpos+wid,0,1),aspect="auto")
    minpos+=wid
    minpos+=0.2
    #plt.annotate(k,xy=(minpos-0.5*wid-0.2,0.5),ha="center",va="center",fontsize=30,bbox=dict(boxstyle="round",fc="w",ec="k"))
    plt.annotate(k,xy=(minpos-wid-0.2+0.09,0.03),ha="left",va="bottom",fontsize=30,color="white")
    #plt.imshow(v,extent=(0.1+i,0.9+i,0,hei),aspect="auto")


plt.axis("off")

#plt.imshow(dic["A"],extent=(0,1,0,heis[0]),aspect="auto")
#plt.imshow(dic["C"],extent=(0,1,heis[0],heis[0]+heis[2]),aspect="auto")
#plt.imshow(dic["B"],extent=(1,2,0,heis[1]),aspect="auto")
#dic={key:val.transpose(1,0,2) for key,val in dic.items()}
#plt.imshow(dic["A"],extent=(0,heis[0],1,2),aspect="auto")
#plt.imshow(dic["C"],extent=(heis[0],heis[0]+heis[2],1,2),aspect="auto")
#plt.imshow(dic["B"],extent=(0,heis[1],0,1),aspect="auto")
print(minpos)
plt.xlim(0,minpos)
plt.ylim(0,1)
plt.axis("off")

#plt.annotate("A",xy=(0.5*heis[0],1.5),ha="center",va="center",fontsize=40)
#plt.annotate("B",xy=(0.5*heis[1],0.5),ha="center",va="center",fontsize=40)
#plt.annotate("C",xy=(heis[0]+0.5*heis[2],1.5),ha="center",va="center",fontsize=40)


plt.tight_layout()






plt.savefig(f"imgs/{name}.png",dpi=600)
plt.savefig(f"imgs/{name}.pdf")
plt.savefig("last.png")

 
 
plt.show()


