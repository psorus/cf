import numpy as np
from plt import *
import matplotlib as mpl

np.random.seed(7)
#np.random.seed(0)

fn="embeddings/metal0_0_0.npz"
#fn="embeddings/pallet0_0_0.npz"
#fn="embeddings/metal0_19_0.npz"
f=np.load(fn)

q=f["emb"]

print(q.shape)


corr=np.corrcoef(q.T)
corr=np.abs(corr)

sortval=np.mean(np.abs(corr),axis=0)
sortidx=np.argsort(sortval)
corr=corr[sortidx,:]
corr=corr[:,sortidx]

print(np.mean(corr>0.8))

#draw as graph. where every corr>0.8 is a link

import networkx as nx
G=nx.Graph()
for i in range(corr.shape[0]):
    G.add_node(i)
for i in range(corr.shape[0]):
    for j in range(i+1,corr.shape[1]):
        if corr[i,j]>0.75:
            G.add_edge(i,j)

#pos = nx.spring_layout(G, k=0.75)
pos = nx.spring_layout(G, k=0.25)


plt.figure(figsize=(10,5))

nx.draw(G,pos,node_shape=mpl.markers.MarkerStyle(marker='o',fillstyle='none'),node_size=150,node_color="black",width=0.5,alpha=0.9)

plt.box(True)
plt.gca().set_axis_on()

legend_elements=[
    mpl.lines.Line2D([0],[0],marker="o",color="black",label="Feature",linestyle="None",markersize=10,fillstyle="none"),
    mpl.lines.Line2D([0],[0],marker="_",color="black",label="Correlation",linestyle="None",markersize=10),
    ]


plt.legend(handles=legend_elements,loc="best",frameon=True,framealpha=0.8)


plt.savefig("imgs/corrograph.png",dpi=600)
plt.savefig("imgs/corrograph.pdf")
plt.savefig("last.png")
plt.show()



#plt.imshow(corr,cmap="seismic",vmin=-1,vmax=1)
#plt.colorbar()
#plt.show()


