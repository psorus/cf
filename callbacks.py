import tensorflow as tf
import keras

from collapse import metrics

class EvaluationCallback(keras.callbacks.Callback):

    def __init__(self, data, encoder, toprint=None):
        self.data = data
        self.encoder = encoder
        if toprint is None:
            toprint = ['trivial_features', '|corr|']
        self.toprint = toprint
        self.logs={}

    def on_epoch_end(self, epoch, logs=None):
        pred=self.encoder.predict(self.data,verbose=0)
        met=metrics(pred)
        outp=""
        for key in self.toprint:
            outp += "{}: {:.4f} ".format(key, met[key])
        for key,val in met.items():
            if key not in self.logs:
                self.logs[key]={}
            self.logs[key][epoch]=val

        print("\nEpoch {}: {}".format(epoch, outp))

class StopAtZeroCallback(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self,epoch,logs=None):
        if logs['loss']<1e-10:
            self.model.stop_training=True
            print("Stopping training at epoch {}".format(epoch))

