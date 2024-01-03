import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
import sys

import json

def gen_triplets(x,i,n=10000):
    """ Generates triplets from x, so that two have the same index in i, and one has not """

    iis=list(set(i))
    itox={ii:np.where(i==ii)[0] for ii in iis}
    #print(np.mean([len(zw) for zw in itox.values()]))
    #exit()

    triplets=[]
    for _ in range(n):
        normal_class=np.random.choice(iis)
        abnormal_class=normal_class
        while abnormal_class==normal_class:
            abnormal_class=np.random.choice(iis)
        base,compare=np.random.choice(itox[normal_class],2,replace=False)
        abnormal=np.random.choice(itox[abnormal_class])
        triplets.append((x[base],x[compare],x[abnormal]))
    return np.array(triplets)


def calc_accuracy(temb,tid,gemb,gid,metric=None):
    temb,tid,gemb,gid=gemb,gid,temb,tid

    if metric is None:
        metric=lambda x,y:np.sum((x-y)**2,axis=-1)
    
    firstmatch=[]
    
    for tt,ti in zip(temb,tid):
        dist=metric(tt,gemb)#(tt-gemb)**2
        while len(dist.shape)>1:
            dist=np.sum(dist,axis=-1)
    
        indice=np.argsort(dist)
        for i,ind in enumerate(indice):
            if gid[ind]==ti:
                firstmatch.append(i)
                break
    
    firstmatch=np.array(firstmatch)
    
    ranks=[1,2,3,4,5,6,7,8,9,10]
    
    ret={}
    for rank in ranks:
        ret[rank]=np.mean(firstmatch<rank)
    return ret


def basemodel(t,outp,mult=1,act="relu",compression=None):
    """creates a basic model, taking the shape of the triplets t and tranforming them into outp values. Returns two models. The one to train, and the one to encode values"""


    if len(t.shape)==3:
        samples,_,features=t.shape
    
        inp=keras.layers.Input(shape=t.shape[2:])
        x=keras.layers.Dense(int(128*mult),activation=act)(inp)
        x=keras.layers.Dense(int(128*mult),activation=act)(x)
        x=keras.layers.Dense(int(128*mult),activation=act)(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)
    else:
        samples,_,dim1,dim2,feat=t.shape
        inp=keras.layers.Input(shape=t.shape[2:])
        x=inp
        if not compression is None:
            x=keras.layers.AveragePooling2D((compression,compression),data_format="channels_last")(x)
        for i in range(3):
            x=keras.layers.Conv2D(int(16*mult),(3,3),activation=act)(x)
            x=keras.layers.Conv2D(int(16*mult),(3,3),activation=act)(x)
            x=keras.layers.MaxPooling2D((2,2))(x)
        x=keras.layers.Flatten()(x)
        x=keras.layers.Dense(int(128*mult),activation=act)(x)
        x=keras.layers.Dense(int(128*mult),activation=act)(x)
        x=keras.layers.Dense(int(128*mult),activation=act)(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)
    
    inp2=keras.layers.Input(shape=t.shape[1:])
    a,b,c=inp2[:,0],inp2[:,1],inp2[:,2]
    a=model(a)
    b=model(b)
    c=model(c)
    a=K.expand_dims(a,axis=1)
    b=K.expand_dims(b,axis=1)
    c=K.expand_dims(c,axis=1)
    q=K.concatenate([a,b,c],axis=1)
    model2=keras.models.Model(inputs=inp2,outputs=q)

    return model,model2

def multimodel(t,outp,count=3):
    """creates multiple basic models, each taking the shape (t.shape->outp), concatenates them and returns this as a new model."""
    m1s=[]
    #m2s=[]
    for _ in range(count):
        m1,m2=basemodel(t,outp)
        m1s.append(m1)
        #m2s.append(m2)
    inp=keras.layers.Input(shape=t.shape[2:])
    #ignore model2, but create a new one using the combined embedding
    feats=[m1(inp) for m1 in m1s]
    x=keras.layers.Concatenate()(feats)
    model=keras.models.Model(inputs=inp,outputs=x)

    inp2=keras.layers.Input(shape=t.shape[1:])
    a,b,c=inp2[:,0],inp2[:,1],inp2[:,2]
    a=model(a)
    b=model(b)
    c=model(c)
    a=K.expand_dims(a,axis=1)
    b=K.expand_dims(b,axis=1)
    c=K.expand_dims(c,axis=1)
    q=K.concatenate([a,b,c],axis=1)
    model2=keras.models.Model(inputs=inp2,outputs=q)

    return model,model2


def trivial_features(x):
    """number features-rank matrix"""
    return x.shape[1]-np.linalg.matrix_rank(x)

def average_correlation(x):
    """calculates the average correlation between features"""
    return np.mean(np.corrcoef(x.T))

def quantile_correlation(x):
    """calculates various quantiles of the distribution of correlations between features"""
    quantiles=[0.1,0.333,0.5,0.666,0.9]
    corr=np.corrcoef(x.T)
    corr=[corr[i,j] for i in range(corr.shape[0]) for j in range(i+1,corr.shape[1])]
    corr=[zw for zw in corr if not np.isnan(zw)]
    corr=np.array(corr)
    return np.quantile(corr,quantiles)

def absolute_quantile_correlation(x):
    """calculates various quantiles of the distribution of absolute correlations between features"""
    quantiles=[0.1,0.333,0.5,0.666,0.9]
    corr=np.corrcoef(x.T)
    corr=[corr[i,j] for i in range(corr.shape[0]) for j in range(i+1,corr.shape[1])]
    corr=[zw for zw in corr if not np.isnan(zw)]
    corr=np.array(corr)
    corr=np.abs(corr)
    return np.quantile(corr,quantiles)

def checkup(x):
    """combine various previous functions"""
    return {"trivial_features":float(trivial_features(x)),
            "average_correlation":float(average_correlation(x)),
            "quantile_correlation":[float(zw) for zw in quantile_correlation(x)],
            "absolute_quantile_correlation":[float(zw) for zw in absolute_quantile_correlation(x)]}

def jeckup(x):
    """json dump checkup"""
    return json.dumps(checkup(x),indent=4)

def dropmodel(t,outp,dropout=0.5):
    """mod of basemodel adding dropout to the output"""


    if len(t.shape)==3:
        samples,_,features=t.shape
    
        inp=keras.layers.Input(shape=t.shape[2:])
        x=keras.layers.Dense(128,activation='relu')(inp)
        x=keras.layers.Dense(128,activation='relu')(x)
        x=keras.layers.Dense(128,activation='relu')(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)
    else:
        samples,_,dim1,dim2,feat=t.shape
        inp=keras.layers.Input(shape=t.shape[2:])
        x=inp
        for i in range(3):
            x=keras.layers.Conv2D(16,(3,3),activation='relu')(x)
            x=keras.layers.Conv2D(16,(3,3),activation='relu')(x)
            x=keras.layers.MaxPooling2D((2,2))(x)
        x=keras.layers.Flatten()(x)
        x=keras.layers.Dense(128,activation='relu')(x)
        x=keras.layers.Dense(128,activation='relu')(x)
        x=keras.layers.Dense(128,activation='relu')(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)
    
    inp2=keras.layers.Input(shape=t.shape[1:])
    a,b,c=inp2[:,0],inp2[:,1],inp2[:,2]
    a=model(a)
    b=model(b)
    c=model(c)
    a=K.expand_dims(a,axis=1)
    b=K.expand_dims(b,axis=1)
    c=K.expand_dims(c,axis=1)
    q=K.concatenate([a,b,c],axis=1)
    q=keras.layers.Dropout(dropout)(q)
    model2=keras.models.Model(inputs=inp2,outputs=q)

    return model,model2



def projmodel(t,outp,mult=1,act="relu",compression=None):
    """creates a basic model, taking the shape of the triplets t and tranforming them into outp values. Returns two models. The one to train, and the one to encode values"""


    if len(t.shape)==3:
        samples,_,features=t.shape
    
        inp=keras.layers.Input(shape=t.shape[2:])
        x=keras.layers.Dense(128*mult,activation=act)(inp)
        x=keras.layers.Dense(128*mult,activation=act)(x)
        x=keras.layers.Dense(128*mult,activation=act)(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)
    else:
        samples,_,dim1,dim2,feat=t.shape
        inp=keras.layers.Input(shape=t.shape[2:])
        x=inp
        if not compression is None:
            x=keras.layers.AveragePooling2D((compression,compression),data_format="channels_last")(x)
        for i in range(3):
            x=keras.layers.Conv2D(16*mult,(3,3),activation=act)(x)
            x=keras.layers.Conv2D(16*mult,(3,3),activation=act)(x)
            x=keras.layers.MaxPooling2D((2,2))(x)
        x=keras.layers.Flatten()(x)
        x=keras.layers.Dense(128*mult,activation=act)(x)
        x=keras.layers.Dense(128*mult,activation=act)(x)
        x=keras.layers.Dense(128*mult,activation=act)(x)
        x=keras.layers.Dense(outp,activation='linear')(x)
        model=keras.models.Model(inputs=inp,outputs=x)

    inpP=keras.layers.Input(shape=x.shape[1:])
    y=inpP
    y=keras.layers.Dense(outp,activation=act)(y)
    y=keras.layers.Dense(outp,activation="linear")(y)
    proj=keras.models.Model(inputs=inpP,outputs=y)
    
    inp2=keras.layers.Input(shape=t.shape[1:])
    a,b,c=inp2[:,0],inp2[:,1],inp2[:,2]
    a=model(a)
    b=model(b)
    c=model(c)
    a=proj(a)
    b=proj(b)
    c=proj(c)
    a=K.expand_dims(a,axis=1)
    b=K.expand_dims(b,axis=1)
    c=K.expand_dims(c,axis=1)
    q=K.concatenate([a,b,c],axis=1)
    model2=keras.models.Model(inputs=inp2,outputs=q)

    return model,model2


class Ensemble(object):
    def __init__(self,models):
        self.models=models
    def fit(self,*args,**kwargs):
        for model in self.models:
            hist=model.fit(*args,**kwargs)
        return hist
    def predict(self,*args,**kwargs):
        pred=[model.predict(*args,**kwargs) for model in self.models]
        return np.concatenate(pred,axis=-1)
    def compile(self,*args,**kwargs):
        for model in self.models:
            model.compile(*args,**kwargs)
    def summary(self):
        print("Ensemble of {} models".format(len(self.models)))



def ensemblemodel(t,outp,count=5,mult=1,act="relu",compression=None):
    """creates a basic model, taking the shape of the triplets t and tranforming them into outp values. Returns two models. The one to train, and the one to encode values"""

    trains,encoders=[],[]
    for i in range(count):
        encoder,train=basemodel(t,outp,mult,act,compression)
        encoders.append(encoder)
        trains.append(train)
    return Ensemble(encoders),Ensemble(trains)

class ParaEnsemble(object):
    def __init__(self,call,count):
        self.call=call
        self.count=count
    def fit_predict(self,*toeval):
        predictions=[[] for i in range(len(toeval))]

        for i in range(self.count):
            [print("!!!!!!!!!!!") for i in range(10)]
            print(i,self.count,i/self.count)
            [print("!!!!!!!!!!!") for i in range(10)]
            model=self.call()
            for j in range(len(toeval)):
                predictions[j].append(model.predict(toeval[j]))
        predictions=[np.concatenate(i,axis=-1) for i in predictions]
        return predictions
            
def paraensemble(t,outp,count=5,*args,**kwargs):
    """creates a basic model, taking the shape of the triplets t and tranforming them into outp values. Returns two models. The one to train, and the one to encode values"""
    def func(training):
        def call():
            pred,train=basemodel(t,outp,*args,**kwargs)
            train.summary()
            training(train)
            return pred
        return ParaEnsemble(call,count)
    return func




from collapse import mac

def retrymodel(t,outp,tries=5,mult=1,act="relu",compression=None):
    """creates a basic model, taking the shape of the triplets t and tranforming them into outp values. Returns two models. The one to train, and the one to encode values"""

    def create_model():
        if len(t.shape)==3:
            samples,_,features=t.shape
        
            inp=keras.layers.Input(shape=t.shape[2:])
            x=keras.layers.Dense(int(128*mult),activation=act)(inp)
            x=keras.layers.Dense(int(128*mult),activation=act)(x)
            x=keras.layers.Dense(int(128*mult),activation=act)(x)
            x=keras.layers.Dense(outp,activation='linear')(x)
            model=keras.models.Model(inputs=inp,outputs=x)
        else:
            samples,_,dim1,dim2,feat=t.shape
            inp=keras.layers.Input(shape=t.shape[2:])
            x=inp
            if not compression is None:
                x=keras.layers.AveragePooling2D((compression,compression),data_format="channels_last")(x)
            for i in range(3):
                x=keras.layers.Conv2D(int(16*mult),(3,3),activation=act)(x)
                x=keras.layers.Conv2D(int(16*mult),(3,3),activation=act)(x)
                x=keras.layers.MaxPooling2D((2,2))(x)
            x=keras.layers.Flatten()(x)
            x=keras.layers.Dense(int(128*mult),activation=act)(x)
            x=keras.layers.Dense(int(128*mult),activation=act)(x)
            x=keras.layers.Dense(int(128*mult),activation=act)(x)
            x=keras.layers.Dense(outp,activation='linear')(x)
            model=keras.models.Model(inputs=inp,outputs=x)
        return model

    anchor=t[:,0,:]
    def calc_mac(model):
        pred=model.predict(anchor)
        return mac(pred)

    models,macs=[],[]
    for i in range(tries):
        model=create_model()
        models.append(model)
        macs.append(calc_mac(model))
    best=np.argmin(macs)
    model=models[best]

    
    inp2=keras.layers.Input(shape=t.shape[1:])
    a,b,c=inp2[:,0],inp2[:,1],inp2[:,2]
    a=model(a)
    b=model(b)
    c=model(c)
    a=K.expand_dims(a,axis=1)
    b=K.expand_dims(b,axis=1)
    c=K.expand_dims(c,axis=1)
    q=K.concatenate([a,b,c],axis=1)
    model2=keras.models.Model(inputs=inp2,outputs=q)

    return model,model2
