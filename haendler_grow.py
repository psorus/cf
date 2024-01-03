import numpy as np
import sys
import os
import json

from collapse import metrics as collapse_metrics
from basemodel import checkup,calc_accuracy


task=0
if len(sys.argv)>1:
    task=int(sys.argv[1])

for task in range(1):

    
    for ds in ["market"]:
    
        parts=[59,69]#current
        parts=[59,69,70,71,72,73,74,75,76,77]#all 1/4
        #parts=[59,69,70,71,72,73,74,75,76,77,58,60,61,62,66]#all of them
        parts=[0]+list(range(101,201,1))

        
        names=[f"embeddings/{ds}_{part}_0.npz" for part in parts]
        names=[zw for zw in names if os.path.exists(zw)]
        
        if len(names)<2:continue
        print(task,ds,"found",len(names),"parts") 
        #if len(names)<len(parts):continue
        
        fs=[np.load(name) for name in names]
        embs,temps,gembs=[f["emb"] for f in fs],[f["temp"] for f in fs],[f["gemb"] for f in fs]
        i,ti,gi=fs[0]["i"],fs[0]["ti"],fs[0]["gi"]
        
        
        emb=np.concatenate(embs,axis=-1)
        gemb=np.concatenate(gembs,axis=-1)
        temp=np.concatenate(temps,axis=-1)
        
        
        
        dic={"task":90,"dataset":ds,"repeat":0,"acc":calc_accuracy(temp,ti,gemb,gi),"checkup":checkup(emb),"collapse":collapse_metrics(emb)}
        
        
        
        np.savez_compressed(f"embeddings/{ds}_90_0.npz",emb=emb,temp=temp,gemb=gemb,i=i,ti=ti,gi=gi,dic=json.dumps(dic,indent=2))
        
        with open(f"results/{ds}_90_0.json","w") as f:
            json.dump(dic,f,indent=2)
        
                                                                                                                                                                                                                                                                                                            
