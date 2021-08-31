# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 07:12:02 2021

@author: user
"""

import pickle
import gp2events
import os

# %% 


with open('data' + os.sep + 'track_events.pickle', 'rb') as handle:
    pieces = pickle.load(handle)

# %% 

# TODO: use the following info properly
# min_pitch = 40
# max_pitch = 94

track_representations = []

for p in pieces:
    print(p.name)
    for i, t in enumerate( p.track_events ):
        print(i)
        tmp_track_repr = gp2events.TrackRepresentation( t, 
                                                       piece_name=p.name,
                                                       track_number=i )
        track_representations.append( tmp_track_repr )

# %% 

with open('data' + os.sep + 'track_representations.pickle', 'wb') as handle:
    pickle.dump(track_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 

with open('data' + os.sep + 'track_representations.pickle', 'rb') as handle:
    b = pickle.load(handle)

# %% string activation

dataset = gp2events.GuitarTabDataset()

for r in b:
    dataset.add_matrices(r)

[x_train, y_train, x_valid, y_valid, x_test, y_test] = dataset.load_data()

string_activation_dataset = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid,
    'x_test': x_test,
    'y_test': y_test,
}

with open('data' + os.sep + 'string_activation_dataset.pickle', 'wb') as handle:
    pickle.dump(string_activation_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 

with open('data' + os.sep + 'string_activation_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)

# %% full tablature

dataset = gp2events.GuitarTabDataset()

for r in b:
    dataset.add_matrices(r)

[x_train, y_train, x_valid, y_valid, x_test, y_test] = dataset.load_data(task='tab')

string_activation_dataset = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid,
    'x_test': x_test,
    'y_test': y_test,
}

with open('data' + os.sep + 'tablature_dataset.pickle', 'wb') as handle:
    pickle.dump(string_activation_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)