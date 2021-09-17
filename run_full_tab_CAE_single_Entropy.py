# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:08:53 2021

@author: user
"""

# check out tensorflow keras constraint for imposing weight constraints

import numpy as np
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
# from tf_agents.utils.common import entropy
from scipy.stats import entropy
# import data_utils

from tensorflow.keras.callbacks import ModelCheckpoint

with open('data' + os.sep + 'full_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train']
y_train = d['y_train']
x_valid = d['x_valid']
y_valid = d['y_valid']
x_test = d['x_test']
y_test = d['y_test']

num_filters = 128
custom_initial_weights = False
tiny_biases = False

# weight constraints
class EntropyMinimizer(tf.keras.constraints.Constraint):
    """Constrains weight tensors to minimum entropy."""
    def __init__(self):
        pass
    
    def __call__(self, w):
        print('w: ', w)
        w_n = tfp.distributions.Categorical( probs=w ).entropy()
        # n = entropy(w_n.T)
        w_n = tf.where( tf.math.is_nan(w_n) , tf.zeros_like(w_n), w_n )
        return tf.math.reduce_sum( w_n )

conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([6, 25, 1], input_shape=[6,25]),
    keras.layers.Conv2D(num_filters, kernel_constraint=EntropyMinimizer(), kernel_size=6, padding='valid', activation='selu'),
])

z = keras.models.Sequential([
    keras.layers.Reshape([20*num_filters]),
    keras.layers.Dense(20*num_filters, activation='selu'),
    keras.layers.Dense(50, activation='selu'),
    keras.layers.Dense(20*num_filters,activation='selu'),
    keras.layers.Reshape([1,20,num_filters])
])

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(1, kernel_size=6, strides=1, padding='valid',
                                 activation='sigmoid', kernel_constraint=EntropyMinimizer(), input_shape=[1,20,num_filters]),
    keras.layers.Reshape([6, 25])
])

# initialise weights
if custom_initial_weights:
    import weights_CAE
    if num_filters == 128:
        w = weights_CAE.get_128_fingering_weights()
    else:
        w = weights_CAE.get_64_fingering_weights()
    b_enc = np.random.rand(num_filters)
    b_dec = np.random.rand(1)
    conv_encoder.set_weights( [w, b_enc] )
    conv_decoder.set_weights( [w, b_dec] )

if tiny_biases:
    b_enc *= 0.0001
    b_dec *= 0.0001

conv_ae = keras.models.Sequential([conv_encoder, z, conv_decoder])

conv_ae.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

filepath = 'data/models/bestValCNN_epoch{epoch:02d}_valAcc{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

filepath_current_best = 'data/models/bestValCNN_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

history = conv_ae.fit(x_train, x_train, validation_data=(x_valid, x_valid), 
                      epochs=1000, batch_size=16, callbacks=[checkpoint, checkpoint_current_best])
