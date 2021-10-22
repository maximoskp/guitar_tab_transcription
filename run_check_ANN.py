import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


# model_path = 'data/models/' + os.listdir('data/models')[0]
model_path = 'data/models/bestValANN_current_best.hdf5'
print('model_path: ', model_path)
model = tf.keras.models.load_model(
    model_path, custom_objects=None, compile=True, options=None
)

with open('data' + os.sep + 'flat_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T

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
    os.makedirs( 'figs/session_ANN_'+str(i), exist_ok=True )
    plt.savefig( 'figs/session_ANN_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )
