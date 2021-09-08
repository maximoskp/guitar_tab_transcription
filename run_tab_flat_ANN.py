# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 23:40:59 2021

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
import data_utils

from tensorflow.keras.callbacks import ModelCheckpoint

with open('data' + os.sep + 'flat_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T


model = keras.models.Sequential()
model.add(keras.layers.Dense(300, activation='selu', input_shape=[x_train.shape[1]]))
model.add(keras.layers.Dense(100, activation='selu'))
model.add(keras.layers.Dense(30, activation='selu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = 'data/models/bestValANN_epoch{epoch:02d}_loss{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

history = model.fit( x_train, y_train, epochs=1000, batch_size=16,
                    validation_data=(x_valid,y_valid), callbacks=[checkpoint])

model.save('models/tab_flat_ANN.h5')

model.evaluate( x_test, y_test )

l = 10
i = np.random.randint(y_test.shape[0]-l)

for j in range(l):
    y_pred = model.predict( [x_test[i+j:i+j+1]] )
    print('predicted: ' + repr(y_pred))
    print('actual: ' + repr(y_test[i+j]))
    print('input: ' + repr( np.where(x_test[i+j, :128] != 0 ) ))
    
    plt.clf()
    plt.subplot(3,1,1)
    # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
    plt.imshow( np.reshape( y_pred[0] , [6, 25]) , cmap='gray_r' )
    plt.title('predicted')
    plt.subplot(3,1,2)
    # plt.bar( np.arange(y_pred.shape[1]) , y_test[i] )
    plt.imshow( np.reshape( y_test[i+j] , [6, 25]) , cmap='gray_r' )
    plt.title('actual')
    plt.subplot(3,1,3)
    plt.bar( np.arange(128) , x_test[i+j:i+j+1, :128][0] )
    plt.title('input')
    os.makedirs( 'figs/session_'+str(i), exist_ok=True )
    plt.savefig( 'figs/session_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )
