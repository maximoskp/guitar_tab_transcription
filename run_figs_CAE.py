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

with open('data' + os.sep + 'full_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train']
y_train = d['y_train']
x_valid = d['x_valid']
y_valid = d['y_valid']
x_test = d['x_test']
y_test = d['y_test']

filepath = 'data/models/bestValCNN_epoch01_valAcc0.99.hdf5'

model = keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)

l = 10
i = np.random.randint(x_test.shape[0]-l)

for j in range(l):
    y_pred = model.predict( [x_test[i+j:i+j+1]] )
    print('predicted: ' + repr(y_pred))
    print('actual: ' + repr(x_test[i+j]))
    
    plt.clf()
    plt.subplot(2,1,1)
    # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
    plt.imshow( y_pred[0] , cmap='gray_r' )
    plt.title('predicted')
    plt.subplot(2,1,2)
    # plt.bar( np.arange(y_pred.shape[1]) , y_test[i] )
    plt.imshow( x_test[i+j] , cmap='gray_r' )
    plt.title('actual')
    os.makedirs( 'figs/session_CAE_'+str(i), exist_ok=True )
    plt.savefig( 'figs/session_CAE_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )
