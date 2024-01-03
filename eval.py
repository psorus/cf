import numpy as np
from plt import *
import json
import os
import sys

task="metal0"
if len(sys.argv)>1:
    task=sys.argv[1]
repetition="0"
if len(sys.argv)>2:
    repetition=sys.argv[2]

def does_fit(x):
    return x.startswith(task) and x.endswith(repetition+".json")

files=os.listdir("results")
files=["results/"+zw for zw in files if does_fit(zw)]

def load_file(fn):
    with open(fn,"r") as f:
        q=json.load(f)
    return q

datas=[load_file(fn) for fn in files]

def print_data(q):
    task,dataset,repeat=q["task"],q["dataset"],q["repeat"]
    acc=q["acc"]
    print(f"{task} ({dataset},{repeat}) :: {acc['1']},{acc['6']},{acc['10']}")

datas.sort(key=lambda x:x["task"])

for q in datas:
    print_data(q)

