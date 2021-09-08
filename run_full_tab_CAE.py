# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:53:34 2021

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

# %% 

with open('data' + os.sep + 'full_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train']
y_train = d['y_train']
x_valid = d['x_valid']
y_valid = d['y_valid']
x_test = d['x_test']
y_test = d['y_test']

# %% 

conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([6, 25, 1], input_shape=[6,25]),
    keras.layers.Conv2D(64, kernel_size=6, padding='same', activation='selu'),
    # keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=6, padding='same', activation='selu'),
    # keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(16, kernel_size=6, padding='valid', activation='selu'),
    # keras.layers.MaxPool2D(pool_size=2)
])

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=6, strides=1, padding='same',
                                 activation='selu', input_shape=[1,20,16]),
    keras.layers.Conv2DTranspose(16, kernel_size=6, strides=1, padding='same',
                                 activation='selu'),
    keras.layers.Conv2DTranspose(1, kernel_size=6, strides=1, padding='valid',
                                 activation='sigmoid'),
    keras.layers.Reshape([6, 25])
])

# %%

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# %% 

conv_ae.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

# %% 

history = conv_ae.fit(x_train, x_train, validation_data=(x_valid, x_valid), 
                      epochs=30, batch_size=32)