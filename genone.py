import sys
import numpy as np
import os

jobs=sys.argv[1:]

files=os.listdir("results")
idents=[zw[:zw.find("_")] for zw in files]
idents=list(set(idents))


with open("sun.sh","w") as f:
    for job in jobs:
        for ident in idents:
            f.write(f"python main.py {ident} {job}\n")

os.system("chmod +x sun.sh")



