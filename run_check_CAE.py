import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


model_path = 'models/' + os.listdir('models')[0]

model = tf.keras.models.load_model(
    model_path, custom_objects=None, compile=True, options=None
)

# %% 

# get encoder and decoder

encoder = model.layers[0]
decoder = model.layers[1]

# %%

# get encoder layer 1 (0 is reshape)
enc1 = encoder.layers[1]

# %% 

enc1w = enc1.weights[0]
# 1 is biases

# %% 

# plot all filters
folder = 'figs/c1/'
for i in range( enc1w.shape[3] ):
    plt.clf()
    plt.imshow(enc1w[:,:,0,i], cmap='gray_r')
    plt.savefig(folder + 'filter_'+str(i)+'.png', dpi=300)