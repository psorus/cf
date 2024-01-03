import numpy as np
import json
from plt import *
import os

def load_one(subset, task, rep, *keys):
    pth=f"results/{subset}_{task}_{rep}.json"
    if not os.path.exists(pth):
        raise ValueError(f"File {pth} does not exist")
    with open(pth, 'r') as f:
        data = json.load(f)
    for key in keys:
        if key not in data:
            raise AttributeError(f"Key {key} not found in {pth}")
        data = data[key]
    return data

def raw_value(subset,task,rep,*keys):
    return load_one(subset,task,rep,*keys)


def load_reps(subset,task,*keys):
    reps = []
    for rep in list(range(10)):
        try:
            reps.append(load_one(subset,task,rep,*keys))
        except ValueError as e:
            pass
    if len(reps)==0:
        raise ValueError(f"No reps found for {subset} {task}")
    return np.mean(reps)

def raw_reps(subset,task,*keys):
    reps = []
    for rep in range(10):
        try:
            reps.append(raw_value(subset,task,rep,*keys))
        except ValueError as e:
            pass
    if len(reps)==0:
        raise ValueError(f"No reps found for {subset} {task}")
    return reps


def access(dataset, task, *keys):
    al=[]
    for ss in [""]+list(range(6)):
        try:
            al.append(load_reps(dataset+str(ss),task,*keys))
        except ValueError as e:
            pass
    if len(al)==0:
        raise ValueError(f"No subsets found for {dataset} {task}")
    if len(al)==1:
        return np.mean(al), 0
        raise ValueError(f"Not enough subsets found for {dataset} {task}")
    return np.mean(al), np.std(al)/np.sqrt(len(al))

def raw_access(dataset, task, *keys):
    al=[]
    for ss in range(6):
        try:
            ac=raw_reps(dataset+str(ss),task,*keys)
            for zw in ac:
                al.append(zw)
        except ValueError as e:
            pass
    if len(al)==0:
        raise ValueError(f"No subsets found for {dataset} {task}")
    return al


def prepare_dic(dataset,dic,*keys):
    #assumes the dic is task:x values, return xvalues->yvalues,std
    x,y,s=[],[],[]
    for task,xx in dic.items():
        try:
            yy,std=access(dataset,task,*keys)
            x.append(xx)
            y.append(yy)
            s.append(std)
        except ValueError as e:
            pass
    if len(x)==0:
        raise ValueError(f"No tasks worked for {dataset}")
    #sort by x
    x,y,s=zip(*sorted(zip(x,y,s)))
    x=np.array(x)
    y=np.array(y)
    s=np.array(s)
    return x,y,s

def range_by_key(key):
    if key=="trivial_features":
        return 0,100
    if key=="not_trivial":
        return 0,100
    return 0,1

def dataset_to_color(dataset):
    if dataset=="metal":
        return "darkgreen"
    if dataset=="pallet":
        return "maroon"
    elif dataset=="market":
        return "darkblue"
    return "grey"




if __name__=="__main__":
    print(access("market","0","acc","1"))






