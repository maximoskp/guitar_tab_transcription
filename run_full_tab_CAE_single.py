# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:08:53 2021

@author: user
"""

import numpy as np
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
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

conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([6, 25, 1], input_shape=[6,25]),
    keras.layers.Conv2D(num_filters, kernel_size=6, padding='valid', activation='selu'),
])

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(1, kernel_size=6, strides=1, padding='valid',
                                 activation='sigmoid', input_shape=[1,20,num_filters]),
    keras.layers.Reshape([6, 25])
])

# initialise weights
import weights_CAE
if num_filters == 128:
    w = weights_CAE.get_128_fingering_weights()
else:
    w = weights_CAE.get_64_fingering_weights()
conv_encoder.set_weights( [w, np.random.rand(num_filters)] )
conv_decoder.set_weights( [w, np.random.rand(1)] )

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

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