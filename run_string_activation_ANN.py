# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 07:14:49 2021

@author: user
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

# %% 

with open('data' + os.sep + 'string_activation_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T


# %% 

model = keras.models.Sequential()
model.add(keras.layers.Dense(300, activation='relu', input_shape=[x_train.shape[1]]))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

# %% 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% 

history = model.fit( x_train, y_train, epochs=30,
                    validation_data=(x_valid,y_valid))

# %% 

model.evaluate( x_test, y_test )

# %% 

i = 7

y_pred = model.predict( [x_test[i:i+1]] )
print('predicted: ' + repr(y_pred))
print('actual: ' + repr(y_test[i]))
print('input: ' + repr( np.where(x_test[i, :128] != 0 ) ))

plt.subplot(3,1,1)
plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
plt.title('predicted')
plt.subplot(3,1,2)
plt.bar( np.arange(y_pred.shape[1]) , y_test[i] )
plt.title('actual')
plt.subplot(3,1,3)
plt.bar( np.arange(128) , x_test[i:i+1, :128][0] )
plt.title('input')