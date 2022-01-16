import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import midi2tab_utils as m2t

with open('..' + os.sep + 'data' + os.sep + 'flat_tablature_rand_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

x_train = d['x_train'].T
y_train = d['y_train'].T
x_valid = d['x_valid'].T
y_valid = d['y_valid'].T
x_test = d['x_test'].T
y_test = d['y_test'].T

# load model
model = keras.models.load_model( '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_current_best.hdf5' )

# model.evaluate( x_test, y_test )

# %%
sessions_num = 10
frames_per_session = 10
session_ids = np.random.choice(y_test.shape[0]-frames_per_session, sessions_num, replace=False)

for i in session_ids:
    for j in range(frames_per_session):
        print('session: ' + str(i) + ' - frame: ' + str(j))
        y_pred = model.predict( [x_test[i+j:i+j+1]] )
        midi = x_test[i+j:i+j+1, :128]
        decision = m2t.midi_and_flat_tab_decision(midi[0], y_pred[0])
        # print('predicted: ' + repr(y_pred))
        # print('actual: ' + repr(y_test[i+j]))
        # print('input: ' + repr( np.where(x_test[i+j, :128] != 0 ) ))
        
        plt.clf()
        fig, ax = plt.subplots(3,1)
        # fig.subplot(3,1,1)
        # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
        ax[0].imshow( np.reshape( y_pred[0] , [6, 25]) , cmap='gray_r' )
        ax[0].set_xticklabels([])
        ax[0].set_ylabel('string')
        ax[0].title.set_text('probabilities')
        # _,ax = plt.subplot(3,1,2)
        # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
        ax[1].imshow( decision , cmap='gray_r' )
        ax[1].set_xticklabels([])
        ax[1].set_ylabel('string')
        ax[1].title.set_text('decision')
        # _,ax = plt.subplot(3,1,3)
        # plt.bar( np.arange(y_pred.shape[1]) , y_test[i] )
        ax[2].imshow( np.reshape( y_test[i+j] , [6, 25]) , cmap='gray_r' )
        ax[2].set_ylabel('string')
        ax[2].set_xlabel('fret')
        ax[2].title.set_text('actual: ' + repr(list(np.where(midi[0])[0])))
        # plt.subplot(2,2,4)
        # plt.bar( np.arange(128) , x_test[i+j:i+j+1, :128][0] )
        # plt.title('input')
        os.makedirs( '../figs/tab_rand_flat_CNN_out/session_'+str(i), exist_ok=True )
        fig.savefig( '../figs/tab_rand_flat_CNN_out/session_'+str(i)+'/fig_'+str(j)+'.png', dpi=300 )


# %% training curves

import pandas as pd

df = pd.read_csv('../models/tab_rand_flat_CNN_out/flat_rand_tab_logger.csv', delimiter=';')

losses = df[['loss','val_loss']]
cossims = df[['cosine_similarity','val_cosine_similarity']]
argmin_epoch = losses['val_loss'].argmin()
print('argmin_epoch:', argmin_epoch)

# save figures
os.makedirs('../figs/training_curves', exist_ok=True)
fig1 = losses.plot().get_figure()
fig1.savefig('../figs/training_curves/' + 'rand_losses.pdf')
fig2 = cossims.plot().get_figure()
fig2.savefig('../figs/training_curves/' + 'rand_cossims.pdf')

original_stdout = sys.stdout
with open('../figs/training_curves/rand_best_epoch.txt', 'w') as f:
    sys.stdout = f
    print(df.iloc[argmin_epoch+1])
    sys.stdout = original_stdout