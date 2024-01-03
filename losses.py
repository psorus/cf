import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
import sys

class aLoss(keras.losses.Loss):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func
    def call(self, y_true, y_pred):
        return self.func(y_true, y_pred)
    def __add__(self, other):
        return aLoss(lambda x,y: self.func(x,y)+other.func(x,y))
    def __sub__(self, other):
        return aLoss(lambda x,y: self.func(x,y)-other.func(x,y))
    def __mul__(self, other):
        if type(other) == aLoss:
            return aLoss(lambda x,y: self.func(x,y)*other.func(x,y))
        else:
            return aLoss(lambda x,y: self.func(x,y)*other)
    def __rmul__(self,other):
        return self.__mul__(other)
    def __str__(self):
        return self.func.__str__()
    def __repr__(self):
        return self.func.__repr__()

def modify_loss_func(func):
    func=aLoss(func)
    return func


def gen_triplet_loss(alpha=0.2):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
        Triplet loss function for TensorFlow.
        
        Arguments:
        y_true -- true labels, required by Keras loss functions but not used in this implementation
        y_pred -- predicted labels or embeddings of shape (batch_size, 3, features)
        alpha -- margin value
        
        Returns:
        loss -- scalar value representing the triplet loss
        """
        
        # Extract the anchor, positive, and negative embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate the Euclidean distance between the anchor and positive embeddings
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Calculate the Euclidean distance between the anchor and negative embeddings
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
        loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
        
        # Compute the mean of the loss over the batch
        loss = tf.reduce_mean(loss)
        
        return loss
    loss_func=modify_loss_func(loss_func)
    return loss_func
    
def gen_logical_loss(alpha=0.2):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
        Triplet loss function for TensorFlow.
        
        Arguments:
        y_true -- true labels, required by Keras loss functions but not used in this implementation
        y_pred -- predicted labels or embeddings of shape (batch_size, 3, features)
        alpha -- margin value
        
        Returns:
        loss -- scalar value representing the triplet loss
        """
        
        # Extract the anchor, positive, and negative embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate the Euclidean distance between the anchor and positive embeddings
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Calculate the Euclidean distance between the anchor and negative embeddings
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
        #loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)

        loss = tf.math.sigmoid(-positive_dist+negative_dist)
        loss = -tf.math.log(loss+1e-8)
        
        # Compute the mean of the loss over the batch
        loss = tf.reduce_mean(loss)
        
        return loss
    loss_func=modify_loss_func(loss_func)
    return loss_func
    
        
    
def gen_singular_loss(alpha=0.2):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Modification of triplet loss, where we dont calculate one loss by a l2 distance, but the average of multiple losses using only one dimension        
        """
        losses=[]
        for feat in range(y_pred.shape[2]):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,feat], axis=1)
            
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss)

            losses.append(loss)
            
        return tf.reduce_mean(losses)
    loss_func=modify_loss_func(loss_func)
    return loss_func
        

def iter_range(size=5,count=3):
    for i in range(count):
        yield slice(i*size,(i+1)*size)

def gen_partial_loss(size=5,count=3,alpha=0.2):#same as singular when count=1; size*count=features assumed
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Modification of triplet loss, where we dont calculate one loss by a l2 distance, but the average of count losses using only size dimension        
        """
        assert size*count==y_pred.shape[2]
        losses=[]
        for slic in iter_range(size,count):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,slic], axis=1)
            
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss)

            losses.append(loss)
            
        return tf.reduce_mean(losses)
    loss_func=modify_loss_func(loss_func)
    return loss_func
        
            
def gen_cross_entropy_loss():
    #that one is stupid. copilot halt
    def loss_func(y_true,y_pred):
        """Instead of triplet loss, use crossentropy. y_true is still not used"""
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        loss = tf.keras.losses.categorical_crossentropy(anchor,positive)
        return tf.reduce_mean(loss)
    loss_func=modify_loss_func(loss_func)
    return loss_func

def gen_zero_mean_loss():
    #mean of all predictions should be zero. Apply on the anchor only
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        return tf.losses.mean_squared_error(tf.reduce_mean(anchor,axis=0),tf.zeros_like(anchor[0]))
    loss_func=modify_loss_func(loss_func)
    return loss_func

def gen_mean_squared_loss():
    #mean of predictions squared should be 1. Apply on the anchor only
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        return tf.losses.mean_squared_error(tf.reduce_mean(tf.square(anchor),axis=0),tf.ones_like(anchor[0]))
    loss_func=modify_loss_func(loss_func)
    return loss_func
                            
def gen_corr_loss():
    #correlation between anchor dimensions should be neglectable
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        anchor=tf.transpose(anchor)
        return tf.reduce_mean(tf.abs(tf.linalg.matmul(anchor,anchor,transpose_a=True)))
    loss_func=modify_loss_func(loss_func)
    return loss_func
                            
def gen_droplet_loss(alpha=0.2,dropout=0.5):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
        Triplet loss function for TensorFlow.
        
        Arguments:
        y_true -- true labels, required by Keras loss functions but not used in this implementation
        y_pred -- predicted labels or embeddings of shape (batch_size, 3, features)
        alpha -- margin value
        
        Returns:
        loss -- scalar value representing the triplet loss
        """

        #mult=np.random.normal(0,1)
        #mult=tf.random.uniform((1,),0,1)
        #print("drawn",mult)

        feature_count=y_pred.shape[-1]
        selector = tf.random.uniform(shape=(feature_count,), minval=0, maxval=1)
        selector = tf.cast(tf.math.less(selector, 1-dropout), tf.float32)
        selector = selector/(1e-6+K.mean(selector))
        y_pred = tf.multiply(y_pred, selector)
        
        # Extract the anchor, positive, and negative embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate the Euclidean distance between the anchor and positive embeddings
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Calculate the Euclidean distance between the anchor and negative embeddings
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
        loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
        
        # Compute the mean of the loss over the batch
        loss = tf.reduce_mean(loss)
        
        return loss#*mult
    loss_func=modify_loss_func(loss_func)
    return loss_func
        
    
def gen_singular_loss(alpha=0.2):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Modification of triplet loss, where we dont calculate one loss by a l2 distance, but the average of multiple losses using only one dimension        
        """
        losses=[]
        for feat in range(y_pred.shape[2]):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,feat], axis=1)
            
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss)

            losses.append(loss)
            
        return tf.reduce_mean(losses)
    loss_func=modify_loss_func(loss_func)
    return loss_func
        

def iter_range(size=5,count=3):
    for i in range(count):
        yield slice(i*size,(i+1)*size)

def gen_partial_loss(size=5,count=3,alpha=0.2):#same as singular when count=1; size*count=features assumed
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Modification of triplet loss, where we dont calculate one loss by a l2 distance, but the average of count losses using only size dimension        
        """
        assert size*count==y_pred.shape[2]
        losses=[]
        for slic in iter_range(size,count):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,slic], axis=1)
            
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss)

            losses.append(loss)
            
        return tf.reduce_mean(losses)
    loss_func=modify_loss_func(loss_func)
    return loss_func
        
            
def gen_cross_entropy_loss():
    #that one is stupid. copilot halt
    def loss_func(y_true,y_pred):
        """Instead of triplet loss, use crossentropy. y_true is still not used"""
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        loss = tf.keras.losses.categorical_crossentropy(anchor,positive)
        return tf.reduce_mean(loss)
    loss_func=modify_loss_func(loss_func)
    return loss_func

def gen_zero_mean_loss():
    #mean of all predictions should be zero. Apply on the anchor only
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        return tf.losses.mean_squared_error(tf.reduce_mean(anchor,axis=0),tf.zeros_like(anchor[0]))
    loss_func=modify_loss_func(loss_func)
    return loss_func

def gen_mean_squared_loss():
    #mean of predictions squared should be 1. Apply on the anchor only
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        return tf.losses.mean_squared_error(tf.reduce_mean(tf.square(anchor),axis=0),tf.ones_like(anchor[0]))
    loss_func=modify_loss_func(loss_func)
    return loss_func
                            
def gen_corr_loss_old():
    #correlation between anchor dimensions should be neglectable
    def loss_func(y_true,y_pred):
        #ypred is (samples,3,features)
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        #anchor is (samples,features)
        #I want for each feature the mean to be zero
        anchor=tf.transpose(anchor)
        return tf.reduce_mean(tf.abs(tf.linalg.matmul(anchor,anchor,transpose_a=True)))
    loss_func=modify_loss_func(loss_func)
    return loss_func
                            
def gen_droplet_loss(alpha=0.2,dropout=0.5):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
        Triplet loss function for TensorFlow.
        
        Arguments:
        y_true -- true labels, required by Keras loss functions but not used in this implementation
        y_pred -- predicted labels or embeddings of shape (batch_size, 3, features)
        alpha -- margin value
        
        Returns:
        loss -- scalar value representing the triplet loss
        """

        #mult=np.random.normal(0,1)
        #mult=tf.random.uniform((1,),0,1)
        #print("drawn",mult)

        feature_count=y_pred.shape[-1]
        selector = tf.random.uniform(shape=(feature_count,), minval=0, maxval=1)
        selector = tf.cast(tf.math.less(selector, 1-dropout), tf.float32)
        selector = selector/(1e-6+K.mean(selector))
        y_pred = tf.multiply(y_pred, selector)
        
        # Extract the anchor, positive, and negative embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate the Euclidean distance between the anchor and positive embeddings
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Calculate the Euclidean distance between the anchor and negative embeddings
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
        loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
        
        # Compute the mean of the loss over the batch
        loss = tf.reduce_mean(loss)
        
        return loss#*mult
    loss_func=modify_loss_func(loss_func)
    return loss_func





def gen_minima_loss(size=5,count=3,alpha=0.2):#same as singular when count=1; size*count=features assumed
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Similar to partial loss, but min instead mean        
        """
        assert size*count==y_pred.shape[2]
        losses=[]
        for slic in iter_range(size,count):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,slic], axis=1)
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss,axis=-1)
            losses.append(loss)
        return tf.reduce_mean(tf.reduce_min(losses,axis=0))
    loss_func=modify_loss_func(loss_func)
    return loss_func
        
def gen_maxima_loss(size=5,count=3,alpha=0.2):#same as singular when count=1; size*count=features assumed
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Similar to partial loss, but max instead mean        
        """
        assert size*count==y_pred.shape[2]
        losses=[]
        for slic in iter_range(size,count):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,slic], axis=1)
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss,axis=-1)
            losses.append(loss)
        return tf.reduce_mean(tf.reduce_max(losses,axis=0))
    loss_func=modify_loss_func(loss_func)
    return loss_func
            
def gen_corr_loss(size=5,count=3,alpha=0.2):#same as singular when count=1; size*count=features assumed
    def loss_func(y_true,y_pred,alpha=alpha):
        """
            Similar to partial loss, but min instead mean        
        """
        assert size*count==y_pred.shape[2]
        losses=[]
        for slic in iter_range(size,count):
            # Extract the anchor, positive, and negative embeddings
            anchor, positive, negative = tf.unstack(y_pred[:,:,slic], axis=1)
            # Calculate the Euclidean distance between the anchor and positive embeddings
            positive_dist = tf.abs(anchor - positive)
            # Calculate the Euclidean distance between the anchor and negative embeddings
            negative_dist = tf.abs(anchor - negative)
            # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
            loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
            # Compute the mean of the loss over the batch
            loss = tf.reduce_mean(loss,axis=-1)
            losses.append(loss)
        products=[]
        for i in range(len(losses)):
            for j in range(i+1,len(losses)):
                products.append(losses[i]*losses[j])
        return tf.reduce_mean(products)
    loss_func=modify_loss_func(loss_func)
    return loss_func

def gen_alternate_loss(alpha=0.2):
    def loss_func(y_true,y_pred,alpha=alpha):
        """
        Triplet loss function for TensorFlow.
        
        Arguments:
        y_true -- true labels, required by Keras loss functions but not used in this implementation
        y_pred -- predicted labels or embeddings of shape (batch_size, 3, features)
        alpha -- margin value
        
        Returns:
        loss -- scalar value representing the triplet loss
        """
        
        # Extract the anchor, positive, and negative embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate the Euclidean distance between the anchor and positive embeddings
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Calculate the Euclidean distance between the anchor and negative embeddings
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        loss = K.log((K.exp(positive_dist)/(K.exp(positive_dist)+K.exp(negative_dist)+0.00000001)))
        
        # Calculate the loss by taking the maximum of (positive_dist - negative_dist + alpha, 0)
        #loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
        
        # Compute the mean of the loss over the batch
        loss = tf.reduce_mean(loss)
        
        return loss
    loss_func=modify_loss_func(loss_func)
    return loss_func
