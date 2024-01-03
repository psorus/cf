import numpy as np
from plt import *

import tensorflow as tf
from tensorflow import keras

import os
import sys


from basemodel import *
from basemodel import gen_triplets,calc_accuracy,basemodel,trivial_features,checkup,jeckup,multimodel,dropmodel,projmodel,ensemblemodel,retrymodel,paraensemble
from losses import gen_triplet_loss,gen_singular_loss,gen_cross_entropy_loss,gen_zero_mean_loss,gen_mean_squared_loss,gen_partial_loss,gen_corr_loss_old,gen_droplet_loss,gen_logical_loss,gen_minima_loss,gen_maxima_loss,gen_corr_loss,gen_alternate_loss

from metrics import gen_usual,gen_minima,gen_maxima

encoding_dim=100 #divisors, 2,4,5,10,20,25,50,100
alpha=1.0

trip_loss=gen_triplet_loss(alpha=alpha)
sing_loss=gen_singular_loss(alpha=alpha)
part_loss1=gen_partial_loss(alpha=alpha,size=10,count=10)
part_loss2=gen_partial_loss(alpha=alpha,size=4,count=25)
part_loss3=gen_partial_loss(alpha=alpha,size=25,count=4)
drop_loss1=gen_droplet_loss(alpha=alpha,dropout=0.1)
drop_loss2=gen_droplet_loss(alpha=alpha,dropout=0.2)
log_loss=gen_logical_loss(alpha=alpha)
min_loss=gen_minima_loss(alpha=alpha,size=25,count=4)
max_loss=gen_maxima_loss(alpha=alpha,size=25,count=4)
corr_loss=gen_corr_loss(alpha=alpha,size=25,count=4)
alt_loss=gen_alternate_loss(alpha=alpha)

mean_punish=gen_zero_mean_loss()
mse_punish=gen_mean_squared_loss()
corr_punish=gen_corr_loss()

punish=mean_punish+mse_punish+corr_punish#missing correlation part

losses={"trip":trip_loss,
        "sing":sing_loss+punish,
        "part1":part_loss1+punish,
        "part2":part_loss2+punish,
        "part3":part_loss3+punish,
        "drop1":drop_loss1,
        "drop2":drop_loss2,
        "log":log_loss,
        "puresing":sing_loss,
        "min":min_loss,
        "max":max_loss,
        "corr":corr_loss,
        "alt":alt_loss}

models1=lambda t: basemodel(t,encoding_dim)
models2=lambda t: multimodel(t,1,count=100)
models3=lambda t: multimodel(t,10,count=10)
models4=lambda t: multimodel(t,4,count=25)
models5=lambda t: multimodel(t,25,count=4)
models6=lambda t: dropmodel(t,encoding_dim,0.5)
models7=lambda t: basemodel(t,encoding_dim*2)
models8=lambda t: basemodel(t,encoding_dim*4)
models9=lambda t: basemodel(t,encoding_dim*8)
modelsA=lambda t: multimodel(t,encoding_dim,4)
modelsB=lambda t: basemodel(t,encoding_dim,2)
modelsC=lambda t: basemodel(t,encoding_dim,4)
modelsD=lambda t: basemodel(t,encoding_dim,8)
modelsE=lambda t: basemodel(t,encoding_dim,act="LeakyReLU")
modelsF=lambda t: basemodel(t,encoding_dim,act="tanh")
modelsG=lambda t: basemodel(t,encoding_dim,act="elu")
modelsH=lambda t: basemodel(t,encoding_dim,compression=2)
modelsI=lambda t: basemodel(t,encoding_dim,compression=4)
modelsJ=lambda t: basemodel(t,encoding_dim,compression=6)
modelsK=lambda t: projmodel(t,encoding_dim)
modelsL=lambda t: basemodel(t,encoding_dim,compression=3)
modelsM=lambda t: basemodel(t,encoding_dim,compression=5)
modelsN=lambda t: basemodel(t,encoding_dim,compression=7)
modelsO=lambda t: basemodel(t,encoding_dim,act="sigmoid")
modelsP=lambda t: basemodel(t,encoding_dim,act="exponential")
modelsQ=lambda t: basemodel(t,encoding_dim,0.25)
modelsR=lambda t: basemodel(t,encoding_dim,0.5)
modelsS=lambda t: basemodel(t,encoding_dim,0.75)
modelsT=lambda t: basemodel(t,encoding_dim,1.5)
modelsU=lambda t: basemodel(t,encoding_dim,2.5)
modelsV=lambda t: multimodel(t,encoding_dim//2,2)
modelsW=lambda t: multimodel(t,encoding_dim//4,4)
modelsX=lambda t: multimodel(t,encoding_dim//5,5)
modelsY=lambda t: multimodel(t,encoding_dim//10,10)
modelsZ=lambda t: multimodel(t,encoding_dim//20,20)
models_1=lambda t: paraensemble(t,encoding_dim//2,2)
models_2=lambda t: paraensemble(t,encoding_dim//4,4)
models_3=lambda t: paraensemble(t,encoding_dim//5,5)
models_4=lambda t: paraensemble(t,encoding_dim//10,10)
models_5=lambda t: paraensemble(t,encoding_dim//20,20)
models_6=lambda t: retrymodel(t,encoding_dim)
models_7=lambda t: retrymodel(t,encoding_dim,tries=50)
models_8=lambda t: retrymodel(t,encoding_dim,tries=100)
models_9=lambda t: paraensemble(t,encoding_dim//3,3)
models_A=lambda t: paraensemble(t,encoding_dim//5,2)
models_B=lambda t: paraensemble(t,encoding_dim//5,3)
models_C=lambda t: basemodel(t,encoding_dim//3)
models_D=lambda t: basemodel(t,encoding_dim//4)
models_E=lambda t: basemodel(t,encoding_dim//5)

models={"1":models1,
        "2":models2,
        "3":models3,
        "4":models4,
        "5":models5,
        "6":models6,
        "7":models7,
        "8":models8,
        "9":models9,
        "A":modelsA,
        "B":modelsB,
        "C":modelsC,
        "D":modelsD,
        "E":modelsE,
        "F":modelsF,
        "G":modelsG,
        "H":modelsH,
        "I":modelsI,
        "J":modelsJ,
        "K":modelsK,
        "L":modelsL,
        "M":modelsM,
        "N":modelsN,
        "O":modelsO,
        "P":modelsP,
        "Q":modelsQ,
        "R":modelsR,
        "S":modelsS,
        "T":modelsT,
        "U":modelsU,
        "V":modelsV,
        "W":modelsW,
        "X":modelsX,
        "Y":modelsY,
        "Z":modelsZ,
        "_1":models_1,
        "_2":models_2,
        "_3":models_3,
        "_4":models_4,
        "_5":models_5,
        "_6":models_6,
        "_7":models_7,
        "_8":models_8,
        "_9":models_9,
        "_A":models_A,
        "_B":models_B,
        "_C":models_C,
        "_D":models_D,
        "_E":models_E}

metricU=gen_usual()
metricMin=gen_minima()
metricMax=gen_maxima()

metrics={"U":metricU,
         "min":metricMin,
         "max":metricMax}



tasks=[]
for loss in ["trip","sing","part1"]:
    for model in ["1","2","3"]:
        tasks.append((loss,model))
tasks.append(("part2","4"))
tasks.append(("part3","5"))

tasks.append(("trip","6"))
tasks.append(("drop1","1"))
tasks.append(("drop2","1"))
tasks.append(("log","1"))
tasks.append(("puresing","1"))
tasks.append(("trip","7"))
tasks.append(("trip","8"))
tasks.append(("trip","9"))
tasks.append(("trip","A"))
tasks.append(("trip","B"))
tasks.append(("trip","C"))
tasks.append(("trip","D"))
tasks.append(("min","1","min"))
tasks.append(("max","1","max"))
tasks.append(("corr","1"))
tasks.append(("min","5","min"))
tasks.append(("max","5","max"))
tasks.append(("corr","5"))
tasks.append(("trip","1","U",{"lr":0.0001}))
tasks.append(("trip","1","U",{"lr":0.01}))
tasks.append(("trip","E"))
tasks.append(("trip","F"))
tasks.append(("trip","G"))
tasks.append(("trip","H"))
tasks.append(("trip","I"))
tasks.append(("trip","J"))
tasks.append(("trip","K"))
tasks.append(("trip","L"))
tasks.append(("trip","M"))
tasks.append(("trip","N"))
tasks.append(("trip","1","U",{"lr":0.002}))
tasks.append(("trip","1","U",{"lr":0.005}))
tasks.append(("trip","1","U",{"lr":0.0002}))
tasks.append(("trip","1","U",{"lr":0.0005}))
tasks.append(("alt","1"))
tasks.append(("trip","O"))
tasks.append(("trip","P"))
tasks.append(("trip","Q"))
tasks.append(("trip","R"))
tasks.append(("trip","S"))
tasks.append(("trip","T"))
tasks.append(("trip","U"))
tasks.append(("trip","V"))
tasks.append(("trip","W"))
tasks.append(("trip","X"))
tasks.append(("trip","Y"))
tasks.append(("trip","Z"))
tasks.append(("trip","_1","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_3","U",{"ens":True}))#for metal gen through number60.py
tasks.append(("trip","_4","U",{"ens":True}))
tasks.append(("trip","_5","U",{"ens":True}))
tasks.append(("trip","_6"))
tasks.append(("trip","_7"))
tasks.append(("trip","_8"))
tasks.append(("trip","_9","U",{"ens":True}))
tasks.append(("trip","_A","U",{"ens":True}))
tasks.append(("trip","_B","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_2","U",{"ens":True}))
tasks.append(("trip","_C","U"))
tasks.append(("trip","_D","U"))
tasks.append(("trip","_E","U"))


tasks=[zw if len(zw)>=3 else (zw[0],zw[1],"U") for zw in tasks]
tasks=[zw if len(zw)>=4 else (zw[0],zw[1],zw[2],{}) for zw in tasks]


#for loss in losses.keys():
#    for model in models.keys():
#        tasks.append((loss,model))

task0=tasks
tasks={i:[losses[loss],models[model],metrics[metric],param] for i,(loss,model,metric,param) in enumerate(tasks)}

if __name__=="__main__":
    print(len(list(tasks.keys())))
    for task in tasks.keys():
        print(task,task0[task])





