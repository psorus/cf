import numpy as np
from plt import *
from vis_tools import access,prepare_dic,range_by_key,dataset_to_color
import sys
import os
import json
from basemodel import calc_accuracy

from collapse import metrics as base_metrics

def metrics(emb,temp,gemb,ti,gi):
    ret=base_metrics(emb)
    print(temp.shape,gemb.shape,ti.shape,gi.shape)
    ret["acc"]=calc_accuracy(temp,ti,gemb,gi)
    print(ret["acc"][1])
    return ret

name=sys.argv[0].split(".")[0]
if name.startswith("plot_"):
    name=name[5:]


dataset="metal"
if len(sys.argv)>1:
    dataset=sys.argv[1]
task="90"
repeat="0"

fns=[f"embeddings/{dataset}{fold}_{task}_{repeat}.npz" for fold in range(10)]
fns=[fn for fn in fns if os.path.exists(fn)]

fs=[np.load(fn) for fn in fns]
embs,temps,gembs,tis,gis=[f["emb"] for f in fs],[f["temp"] for f in fs],[f["gemb"] for f in fs],[f["ti"] for f in fs],[f["gi"] for f in fs]
i,ti,gi=fs[0]["i"],fs[0]["ti"],fs[0]["gi"]
#print(i.shape,ti.shape,gi.shape,embs[0].shape,temps[0].shape,gembs[0].shape)
#exit()

maxima=max((emb.shape[1] for emb in embs))


def avg_dic(*dics):
    flipped={key:[] for key in dics[0].keys()}
    for dic in dics:
        for key,value in dic.items():
            flipped[key].append(value)
    return {key:[np.mean(value),np.std(value)/np.sqrt(len(value))] if not type(value[0]) is dict else avg_dic(*value) for key,value in flipped.items()}

def metrics_by_size(size):
    emb=[emb[:,:size] for emb in embs]
    temp=[temp[:,:size] for temp in temps]
    gemb=[gemb[:,:size] for gemb in gembs]
    return avg_dic(*[metrics(emb[i],temp[i],gemb[i],tis[i],gis[i]) for i in range(len(emb))])


def multiaccess(arr,*keys):
    for key in keys:
        try:
            arr=arr[key]
        except:
            print("key",key,"not found in",arr.keys())
            exit()
    return arr


attributes={"rank1":["acc",1],"rank10":["acc",10],"corr":["|corr|"],"trivial_features":["trivial_features"]}


toshow="rank1"
sizes=list(range(25,maxima+1,25))#+[maxima]
info={size:metrics_by_size(size) for size in sizes}

with open(f"storage/ens_{dataset}.json","w") as f:
    json.dump({"x":sizes,"y":info},f,indent=2)








