# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 07:24:24 2021

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

num_filters = 128
custom_initial_weights = True
binary_initial_weights = True
tiny_biases = False
zero_biases = True
z_is_diagonal = True
z_is_ones = False
z_zero_biases = True

# %% import data
with open('data' + os.sep + 'full_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train']
y_train = d['y_train']
x_valid = d['x_valid']
y_valid = d['y_valid']
x_test = d['x_test']
y_test = d['y_test']

# %% initialize model
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([6, 25, 1], input_shape=[6,25]),
    keras.layers.Conv2D(num_filters, kernel_size=6, padding='valid', activation='selu'),
])
fretboard_extension = 20
z = keras.models.Sequential([
    keras.layers.Reshape([fretboard_extension*num_filters]),
    keras.layers.Dense(fretboard_extension*num_filters, activation='selu'),
    keras.layers.Reshape([1,fretboard_extension,num_filters])
])
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(1, kernel_size=6, strides=1, padding='valid',
                                 activation='sigmoid', input_shape=[1,fretboard_extension,num_filters]),
    keras.layers.Reshape([6, 25])
])

# %% initialize weights
if custom_initial_weights:
    import weights_CAE
    if num_filters == 128:
        if binary_initial_weights:
            w = weights_CAE.get_128_random_binary_row()
        else:
            w = weights_CAE.get_128_fingering_weights()
    else:
        if binary_initial_weights:
            w = weights_CAE.get_64_random_binary_row()
        else:
            w = weights_CAE.get_64_fingering_weights()
    if zero_biases:
        b_enc = np.zeros(num_filters)
        b_dec = np.zeros(1)
    else:
        b_enc = np.random.rand(num_filters)
        b_dec = np.random.rand(1)
    if tiny_biases:
        b_enc *= 0.0001
        b_dec *= 0.0001
    conv_encoder.set_weights( [w, b_enc] )
    conv_decoder.set_weights( [w, b_dec] )
    if z_is_diagonal:
        w_z = np.zeros( (num_filters*fretboard_extension, num_filters*fretboard_extension) )
        np.fill_diagonal( w_z , 1 )
    elif z_is_ones:
        w_z = np.ones( (num_filters*fretboard_extension, num_filters*fretboard_extension) )
    else:
        w_z = np.random.rand( num_filters*fretboard_extension, num_filters*fretboard_extension )
    if z_zero_biases:
        b_z = np.zeros( num_filters*fretboard_extension )
    else:
        b_z = np.random.rand( num_filters*fretboard_extension )

# %% finalize models first
conv_ae = keras.models.Sequential([conv_encoder, z, conv_decoder])
# conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# %% apply filters and weights after finilizing model
conv_encoder.set_weights( [w, b_enc] )
z.layers[1].set_weights( [w_z,b_z])
conv_decoder.set_weights( [w, b_dec] )

# %% examine encoder output
i = 100
x = x_train[i:i+1, :,:]
enc_y = conv_encoder( x ).numpy()
plt.clf()
plt.subplot(2,1,1)
plt.imshow( x[0] , cmap='gray_r' )
plt.subplot(2,1,2)
plt.imshow( enc_y[0,0,:,:], cmap='gray_r' )

# %% verify filters
import CAE_filter_examiner as cfe

enc_CNN_layer = conv_encoder.layers[1]
enc_filters = cfe.get_filters_of_layer(enc_CNN_layer)

dec_CNN_layer = conv_decoder.layers[0]
dec_filters = cfe.get_filters_of_layer(dec_CNN_layer)

# %% test input
for j in range(100):
    i = 900 + j
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
    plt.pause(0.1)