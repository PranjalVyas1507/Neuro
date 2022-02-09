'''
Distiller class for Keras models
'''

from tensorflow import keras
import numpy as np
import tensorflow as tf

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = keras.metrics.Mean(name='distillation_loss')

    @property
    def metrics(self):
        return super().metrics + [self.loss_tracker]

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None) :
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics)    