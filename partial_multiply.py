import os

from datasets import datasets
from tasks import tasks

import sys
tasks=sys.argv[1:]

with open("sun.sh","w") as f:
    for task in tasks:
        for dataset in datasets.keys():
            f.write(f"python3 main.py {dataset} {task}\n")
    #for dataset in datasets.keys():
    #    for task in tasks:
    #        f.write(f"python3 main.py {dataset} {task}\n")

os.system("chmod +x sun.sh")


