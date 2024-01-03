import numpy as np
from plt import *

import tensorflow as tf
from tensorflow import keras

import os
import sys

import json


from basemodel import *
from basemodel import gen_triplets,calc_accuracy,basemodel,trivial_features,checkup,jeckup,multimodel
from losses import gen_triplet_loss,gen_singular_loss,gen_cross_entropy_loss,gen_zero_mean_loss,gen_mean_squared_loss,gen_partial_loss

from datasets import datasets
from tasks import tasks
import time

from callbacks import EvaluationCallback,StopAtZeroCallback
from collapse import metrics as collapse_metrics


debug=False
if "--debug" in sys.argv:
    debug=True
    sys.argv.remove("--debug")

rerun=False
if "--rerun" in sys.argv:
    rerun=True
    sys.argv.remove("--rerun")

dataset=sys.argv[1]
task=int(sys.argv[2])

loss,model,metric,param=tasks[task if task<100 else 0]

ensemblemode=False
if "ens" in param and param["ens"]:
    ensemblemode=True
    print("Entering ensemble mode")

ident=dataset+'_'+str(task)
if not debug:
    print("Identifyier:",ident)
    #time.sleep(3)

#repeats=3
repeats=1
repeats=5

#fn="datas/std/emb0.npz"
#fn="/global/splits/light_on_0.npz"
#fn="/global/metal/fold0.npz"
fn=datasets[dataset]

f=np.load(fn)
try:
    x,tx,gx,idd,tid,gid=f['emb'],f['temb'],f['gemb'],f['id'],f['tid'],f['gid']
except:
    x,tx,gx,idd,tid,gid=f["x"],f["tx"],f["gx"],f["i"],f["ti"],f["gi"]
f.close()

os.makedirs("results",exist_ok=True)
os.makedirs("embeddings",exist_ok=True)


for repeat in range(repeats):
    
    if os.path.isfile(f"results/{ident}_{repeat}.json") and not debug and not rerun:continue
    t=gen_triplets(x,idd,n=10000)
    
    
    #encoder,trainer=basemodel(t,15)
    lr=0.001
    if "lr" in param:
        lr=param["lr"]
    if ensemblemode==False:
        encoder,trainer=model(t)
        
        encoder.summary()
        
        trainer.summary()
        
        trainer.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=loss)

        evi=EvaluationCallback(x,encoder)
    
        callb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True),StopAtZeroCallback()]
        if not "nsemble" in str(encoder):
            callb.append(evi)
        history=trainer.fit(t,t,epochs=100,batch_size=128,shuffle=True,validation_split=0.1,callbacks=callb)
    
        temp,gemb=encoder.predict(tx),encoder.predict(gx)
        emb=encoder.predict(x)

    else:
        def train(model):
            global history
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=loss)
            callb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True),StopAtZeroCallback()]

            history=model.fit(t,t,epochs=100,batch_size=128,shuffle=True,validation_split=0.1,callbacks=callb)
        #print(help(model))
        model=model(t)(train)
        #print(help(model))
        #print(model)
        temp,gemb,emb=model.fit_predict(tx,gx,x)
    
    
    newacc=calc_accuracy(temp,tid,gemb,gid)
    usacc=calc_accuracy(temp,tid,gemb,gid,metric)
    specialized=str(newacc[1]>usacc[1])

    dic={}
    dic={"task":task,"dataset":dataset,"repeat":repeat,"acc":newacc,"checkup":checkup(emb),"history":history.history,"specialized":usacc,"collapse":collapse_metrics(emb),"timecollapse":0}
    try:
        dic["timecollapse"]=evi.logs
    except:
        pass


    if not debug:
        np.savez_compressed("embeddings/"+ident+"_"+str(repeat)+".npz",emb=emb,temp=temp,gemb=gemb,i=idd,ti=tid,gi=gid,dic=json.dumps(dic,indent=2))
        with open(f"results/{ident}_{repeat}.json","w") as f:
            json.dump(dic,f,indent=2)
    else:
        sdic={key:val for key,val in dic.items() if key not in ["history"]}
        print(json.dumps(sdic,indent=2))

    
    #plt.figure(figsize=(10,4))
    #plt.hist(emb.flatten(),bins=100)
    #plt.axvline(np.mean(emb),color='r')
    #plt.axvspan(np.mean(emb)-np.std(emb),np.mean(emb)+np.std(emb),color='r',alpha=0.2)
    #plt.savefig("last.png")
    #plt.show()
    
    
                                        
