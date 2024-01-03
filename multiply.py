import os

from datasets import datasets
from tasks import tasks


with open("run.sh","w") as f:
    for task in tasks.keys():
        for dataset in datasets.keys():
            f.write(f"python3 ~/handlesubmit.py python3 main.py {dataset} {task}\n")

os.system("chmod +x run.sh")


