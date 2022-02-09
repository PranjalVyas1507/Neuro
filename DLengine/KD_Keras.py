'''
Distiller class for Keras models
'''

from curses import termattrs
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

    def compile(self, optimizer, distillation_fn, temperature = 10, loss=None, metrics=metrics, loss_weights=None,
     weighted_metrics=None) :
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics)
        self. distillation_fn = distillation_fn
        self.temperature = temperature
    
    def train_step(self, data):
        X, y = data

        #Teacher model prediction forward pass
        teacher_predictions = self.teacher(X, training= False)

        with tf.GradientTape as tape:
            #Student Forward pass
            student_predictions = self.student(X, training=True)

            #Compute distillation loss
            distillation_loss = self.distillation_fn(
                tf.nn.softmax(student_predictions/self.temperature, axis = 1),
                tf.nn.softmax(teacher_predictions/self.temperature, axis =1 ),
            )

        #Compute gradients
        grads = tape.gradient(distillation_loss, self.student.trainable_variables)

        #Update weights
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        #Report progress and update loss state
        self.loss_tracker.update_state(distillation_loss)

    def test_step(self, data):
        X,y = data
        student_predictions = self.student(X, training=False)
        teacher_predictions = self.teacher(X, training=False)

        #Compute distillation loss
        distillation_loss = self.distillation_fn(
            tf.nn.softmax(student_predictions/self.temperature, axis = 1),
            tf.nn.softmax(teacher_predictions/self.temperature, axis =1 ),
        )

        #Report progress and update loss state
        