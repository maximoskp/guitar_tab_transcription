import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

with open('..' + os.sep + 'data' + os.sep + 'flat_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T

# load model
model = keras.models.load_model( '../models/tab_flat_CNN_out/tab_flat_CNN_out_current_best.hdf5' )

model.evaluate( x_test, y_test )

# %%
sessions_num = 10
session_ids = np.random.choice(y_test.shape[0]-l, sessions_num, replace=False)

frames_per_session = 10

for i in session_ids:
    for j in range(frames_per_session):
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
        os.makedirs( '../figs/tab_flat_CNN_out/session_'+str(i), exist_ok=True )
        plt.savefig( '../figs/tab_flat_CNN_out/session_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )
