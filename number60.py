import numpy as np
import sys
import os
import json

from collapse import metrics as collapse_metrics
from basemodel import checkup,calc_accuracy


task=0
if len(sys.argv)>1:
    task=int(sys.argv[1])


parts=[67,68]

names=[f"embeddings/metal{task}_{part}_0.npz" for part in parts]

assert all([os.path.exists(name) for name in names])

fs=[np.load(name) for name in names]
embs,temps,gembs=[f["emb"] for f in fs],[f["temp"] for f in fs],[f["gemb"] for f in fs]
i,ti,gi=fs[0]["i"],fs[0]["ti"],fs[0]["gi"]


emb=np.concatenate(embs,axis=-1)
gemb=np.concatenate(gembs,axis=-1)
temp=np.concatenate(temps,axis=-1)



dic={"task":60,"dataset":"metal"+str(task),"repeat":0,"acc":calc_accuracy(temp,ti,gemb,gi),"checkup":checkup(emb),"collapse":collapse_metrics(emb)}



np.savez_compressed(f"embeddings/metal{task}_60_0.npz",emb=emb,temp=temp,gemb=gemb,i=i,ti=ti,gi=gi,dic=json.dumps(dic,indent=2))

with open(f"results/metal{task}_60_0.json","w") as f:
    json.dump(dic,f,indent=2)


