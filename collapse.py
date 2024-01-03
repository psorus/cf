
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
import sys

import json

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

def average_absolute(x):
    return np.mean(np.abs(np.corrcoef(x.T)))

mac=average_absolute    

def twothirds(x):
    return absolute_quantile_correlation(x)[-2]

def de_nanned_average(x):
    cc=np.corrcoef(x.T) 
    cc=cc.flatten()
    cc=[zw if not np.isnan(zw) else 1.0 for zw in cc]
    return np.mean(np.abs(cc))

def metrics(x):
    return {"trivial_features":float(trivial_features(x)),
            "|corr|":float(average_absolute(x)),
            "2/3":float(twothirds(x)),
            "nanprice":float(de_nanned_average(x))}



