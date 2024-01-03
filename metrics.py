import numpy as np

def sumup(q):
    """while more than 1d, sum last dim"""
    while q.ndim > 1:
        q = np.sum(q,axis=-1)
    return q


def gen_usual():
    def loss(a,b):
        return sumup(np.square(a-b))
    return loss

def gen_minima(size=25,count=4):
    """usual loss for each size subset, minimum of those"""
    usual=gen_usual()
    def loss(a,b):
        lss=[]
        for i in range(count):
            a=a[i*size:(i+1)*size]
            b=b[:,i*size:(i+1)*size]
            lss.append(usual(a,b))
        return np.min(lss,axis=0)
    return loss

def gen_maxima(size=25,count=4):
    """usual loss for each size subset, maximum of those"""
    usual=gen_usual()
    def loss(a,b):
        lss=[]
        for i in range(count):
            a=a[i*size:(i+1)*size]
            b=b[:,i*size:(i+1)*size]
            lss.append(usual(a,b))
        return np.max(lss,axis=0)
    return loss


