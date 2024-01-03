import numpy as np
import sys
import os
import json

from collapse import metrics as collapse_metrics
from basemodel import checkup,calc_accuracy


task=78
if len(sys.argv)>1:
    task=int(sys.argv[1])

goto=66
if len(sys.argv)>2:
    goto=int(sys.argv[2])

count=3
if len(sys.argv)>3:
    count=int(sys.argv[3])

#78->66  3
#79->59  4
#80->60  5



parts=[i for i in range(count)]

names=[f"embeddings/market_{task}_{part}.npz" for part in parts]

assert all([os.path.exists(name) for name in names])

fs=[np.load(name) for name in names]
embs,temps,gembs=[f["emb"] for f in fs],[f["temp"] for f in fs],[f["gemb"] for f in fs]
i,ti,gi=fs[0]["i"],fs[0]["ti"],fs[0]["gi"]


emb=np.concatenate(embs,axis=-1)
gemb=np.concatenate(gembs,axis=-1)
temp=np.concatenate(temps,axis=-1)



dic={"task":goto,"dataset":"market","repeat":0,"acc":calc_accuracy(temp,ti,gemb,gi),"checkup":checkup(emb),"collapse":collapse_metrics(emb)}



np.savez_compressed(f"embeddings/market_{goto}_0.npz",emb=emb,temp=temp,gemb=gemb,i=i,ti=ti,gi=gi,dic=json.dumps(dic,indent=2))

with open(f"results/market_{goto}_0.json","w") as f:
    json.dump(dic,f,indent=2)


