# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 07:12:02 2021

@author: user
"""

# RUN
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os

# flat tablature dataset
dataset = data_utils.GuitarTabDataset(task='full_tablature', 
                                     output_representation='full_tablature',
                                     history=2)

rep_folder = 'data/track_representation_parts'
reps = os.listdir( rep_folder )

for f_name in reps:
    f_path = os.path.join( rep_folder, f_name )
    b = None
    print(f_path)
    with open(f_path, 'rb') as handle:
        b = pickle.load(handle)
    for r in b:
        dataset.add_matrices(r)

# create final matrices

[x_train, y_train, x_valid, y_valid, x_test, y_test] = dataset.load_full_tabs()

full_tablature_dataset = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid,
    'x_test': x_test,
    'y_test': y_test,
}

with open('data' + os.sep + 'full_tablature_dataset.pickle', 'wb') as handle:
    pickle.dump(full_tablature_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
