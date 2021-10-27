# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:42:37 2021

@author: user
"""

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

num_filters = 64

# %%
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([6, 25, 1], input_shape=[6,25]),
    keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', activation='selu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(num_filters//2, kernel_size=3, padding='same', activation='selu'),
    keras.layers.MaxPooling2D(pool_size=2)
])

'''
z = keras.models.Sequential([
    keras.layers.Reshape([6*num_filters//2]),
    # keras.layers.Dense(6*num_filters//2, activation='selu'),
    # keras.layers.Dense(40, activation='selu'),
    keras.layers.Dense(6*num_filters//2,activation='selu'),
    keras.layers.Reshape([1,6,num_filters//2])
])
'''

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(num_filters//2, kernel_size=3, strides=2, padding='valid',
                                 activation='selu', input_shape=[1,6,num_filters//2]),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same',
                                 activation='selu'),
    # keras.layers.Lambda(lambda x: x[:,:,:-1,:]),
    # keras.layers.Reshape([6, 25])
])

out_layer = keras.models.Sequential([
    keras.layers.Lambda(lambda x: x[:,:,:-1,:]),
    keras.layers.Reshape([6, 25])
])

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder, out_layer])

conv_ae.compile(loss='mean_squared_error', optimizer='adam',
                metrics=['cosine_similarity'])

conv_ae.summary()

# %% 

filepath = 'data/models/bestValCNN_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = 'data/models/bestValCNN_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

history = conv_ae.fit(x_train, x_train, validation_data=(x_valid, x_valid), 
                      epochs=1000, batch_size=16, callbacks=[checkpoint, checkpoint_current_best])


# %% test input
for j in range(100):
    i = 1200 + j
    x = x_train[i:i+1, :,:]
    y_out = conv_ae( x )
    y = y_out.numpy()
    
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow( x[0] , cmap='gray_r' )
    plt.title('in')
    plt.subplot(2,1,2)
    plt.imshow( y[0] , cmap='gray_r' )
    plt.title('out')
    plt.pause(0.5)