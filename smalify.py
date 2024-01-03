import numpy as np
import os
from tqdm import tqdm

base="datas"
folders=os.listdir(base)
files=[]
for folder in folders:
    for file in os.listdir(base+"/"+folder):
        if not file.endswith(".npz"):continue
        files.append(base+"/"+folder+"/"+file)

for folder in folders:
    if not os.path.isdir("smalldata/"+folder):os.mkdir("smalldata/"+folder)

def smoulder(q):
    q=q[:5]
    return q


for file in tqdm(files):
    outp=file.replace(base+"/","smalldata/")
    
    f=np.load(file,allow_pickle=True)
    keys=list(f.keys())
    dic={key:smoulder(f[key]) for key in keys}

    np.savez_compressed(outp,**dic)



