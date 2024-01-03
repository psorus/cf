import numpy as np
from plt import *
import json
import os
import sys

tasks=["std","metal","pallet","market"]
if len(sys.argv)>1:
    tasks=sys.argv[1:]
    
if os.path.isfile("gun.sh"):
    os.remove("gun.sh")


for task in tasks:
    print("TASK:",task)
    theTask=task
    
    def does_fit(x):
        return x.startswith(task) and x.endswith(".json")
    
    files=os.listdir("results")
    files=["results/"+zw for zw in files if does_fit(zw)]
    
    def load_file(fn):
        with open(fn,"r") as f:
            q=json.load(f)
        return q
    
    datas=[load_file(fn) for fn in files]
    #acc=[zw["acc"] for zw in datas]
    #tasks=[zw["task"] for zw in datas]
    #datasets=[zw["dataset"] for zw in datas]
    
    
    by_dataset={}
    for zw in datas:
        if not zw["dataset"] in by_dataset:
            by_dataset[zw["dataset"]]=[]
        by_dataset[zw["dataset"]].append(zw)
    
    def largest_common_subset(lis):
        lis=list(lis)
        ret=[]
        for val in lis[0]:
            if all([val in zw for zw in lis]):
                ret.append(val)
        return ret
    
    
    task_by_dataset={key:sorted(list(set([zw["task"] for zw in by_dataset[key]]))) for key in sorted(list(by_dataset.keys()))}
    considered=largest_common_subset(task_by_dataset.values())
    missing=list(set([zw["task"] for zw in datas])-set(considered))
    if len(missing)>0:
        print(task_by_dataset)
        print(considered)
        print("Missing",missing)
    with open("gun.sh","a") as f:
        for task in missing:
            for dataset in by_dataset:
                found=False
                for zw in by_dataset[dataset]:
                    if zw["task"]==task:
                        found=True
                if not found:
                    print(f"python3 main.py {dataset} {task}")
                    f.write(f"python3 main.py {dataset} {task}\n")
    
    def average(*dics):
        keys=dics[0].keys()
        count=len(dics)
        return {key:sum([dic[key] for dic in dics])/count for key in keys}
    
    def standart_deviation(*dics):
        keys=dics[0].keys()
        count=len(dics)
        return {key:np.std([dic[key] for dic in dics])/np.sqrt(count) for key in keys}
    
    def selector(q):
        ret={key:val for key,val in q["acc"].items()}
        #ret["quantile"]=q["checkup"]["absolute_quantile_correlation"][-1]
        ret["quantile"]=q["collapse"]["|corr|"]
        ret["useless"]=q["collapse"]["trivial_features"]
        return ret


    def handle_dataset(dataset):
        meas=by_dataset[dataset]
        meas=[zw for zw in meas if zw["task"] in considered]
        by_task={cons:[] for cons in considered}
        for zw in meas:
            by_task[zw["task"]].append(zw)
        ret={}
        for task in considered:
            reps=by_task[task]
            accs=[selector(zw) for zw in reps]
            acc=average(*accs)
            ret[task]=acc
        return ret
    
    def pprint_simple(task,dic):
        acc1,acc6,acc10=dic["1"],dic["6"],dic["10"]
        print(f"{task} :: {acc1},{acc6},{acc10}")
    def pprint(task,dic,std=None):
        if std is None:
            return pprint_simple(task,dic)
        acc1,acc6,acc10=dic["1"],dic["6"],dic["10"]
        std1,std6,std10=std["1"],std["6"],std["10"]
        q_acc,q_std=dic["quantile"],std["quantile"]
        u_acc,u_std=dic["useless"],std["useless"]
        print(f"{task} :: {acc1:.3f}+-{std1:.3f}    {acc6:.3f}+-{std6:.3f}    {acc10:.3f}+-{std10:.3f}    ({q_acc:.3f}+-{q_std:.3f} , {u_acc:.1f}+-{u_std:.1f})")

        return {"acc":acc1,"q":q_acc,"u":u_acc}
    
    
    infos={dataset:handle_dataset(dataset) for dataset in by_dataset}
    infos={task:{dataset:infos[dataset][task] for dataset in infos} for task in considered}
    
    mean={task:average(*[infos[task][dataset] for dataset in infos[task]]) for task in considered}
    std={task:standart_deviation(*[infos[task][dataset] for dataset in infos[task]]) for task in considered}
    requiem={}
    for task in considered:
        ac=pprint(task,mean[task],std[task])
        for key,val in ac.items():
            if not key in requiem:
                requiem[key]=[]
            requiem[key].append(val)
    with open(f"merges/{theTask}.json","w") as f:
        json.dump(requiem,f,indent=2)
    
os.system("chmod +x gun.sh")
    
    
