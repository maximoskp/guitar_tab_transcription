# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 07:30:57 2021

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
sys.path.insert(1, '..')
import data_utils

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

with open('..' + os.sep + 'data' + os.sep + 'flat_tablature_rand_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T

num_filters = 128
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
    keras.layers.Reshape([6*25]),
    keras.layers.Dense(y_train.shape[1], activation='sigmoid')
])

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='selu', input_shape=[x_train.shape[1]]))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='selu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(6*num_filters//2, activation='selu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())
# to apply lstm, timesteps need to be keept in the input
# model.add(keras.layers.LSTM(6*num_filters//2, activation='selu'))
model.add(keras.layers.Reshape([1,6,num_filters//2]))
model.add(conv_decoder)
model.add(out_layer)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['cosine_similarity'])
model.summary()

os.makedirs( 'models/tab_rand_flat_CNN_out', exist_ok=True )

# %% 
filepath = '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

csv_logger = CSVLogger('../models/tab_rand_flat_CNN_out/flat_rand_tab_logger.csv', append=True, separator=';')

history = model.fit( x_train, y_train, epochs=1000, batch_size=16,
                    validation_data=(x_valid,y_valid), callbacks=[checkpoint, checkpoint_current_best, csv_logger])

# model.save('models/tab_flat_ANN.h5')

# model.evaluate( x_test, y_test )

# # %%
# sessions_num = 10
# session_ids = np.random.choice(y_test.shape[0]-l, sessions_num, replace=False)

# frames_per_session = 10

# for i in session_ids:
#     for j in range(frames_per_session):
#         y_pred = model.predict( [x_test[i+j:i+j+1]] )
#         print('predicted: ' + repr(y_pred))
#         print('actual: ' + repr(y_test[i+j]))
#         print('input: ' + repr( np.where(x_test[i+j, :128] != 0 ) ))
        
#         plt.clf()
#         plt.subplot(3,1,1)
#         # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
#         plt.imshow( np.reshape( y_pred[0] , [6, 25]) , cmap='gray_r' )
#         plt.title('predicted')
#         plt.subplot(3,1,2)
#         # plt.bar( np.arange(y_pred.shape[1]) , y_test[i] )
#         plt.imshow( np.reshape( y_test[i+j] , [6, 25]) , cmap='gray_r' )
#         plt.title('actual')
#         plt.subplot(3,1,3)
#         plt.bar( np.arange(128) , x_test[i+j:i+j+1, :128][0] )
#         plt.title('input')
#         os.makedirs( 'figs/tab_flat_CNN_out/session_'+str(i), exist_ok=True )
#         plt.savefig( 'figs/tab_flat_CNN_out/session_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )
