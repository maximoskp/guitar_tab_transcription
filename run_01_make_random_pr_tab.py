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

# RUN to save first time
track_event_files = os.listdir( 'data/dadaGP_event_parts' )

excepted2_pieces = []

os.makedirs( 'data/track_representation_withRandomness_parts', exist_ok=True )

for te_file in track_event_files:
    with open('data/dadaGP_event_parts' + os.sep + te_file, 'rb') as handle:
        pieces = pickle.load(handle)
    track_representations = []
    for p in pieces:
        print(p.name)
        excepted = False
        for i, t in enumerate( p.track_events ):
            print(i)
            try:
                tmp_track_repr = data_utils.TrackRepresentation( t, 
                                                               piece_name=p.name,
                                                               track_number=i,
                                                               random_pr=0.5 )
            except:
                excepted2_pieces.append( p.name )
                excepted = True
            if not excepted:
                track_representations.append( tmp_track_repr )
    tmp_array = te_file.split('.')[0].split('_')[:-1]
    tmp_array.append('representations.pickle')
    rep_name = '_'.join(tmp_array)
    with open('data/track_representation_withRandomness_parts' + os.sep + rep_name, 'wb') as handle:
        pickle.dump(track_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'excepted2Rand_pieces.pickle', 'wb') as handle:
    pickle.dump(excepted2_pieces, handle, protocol=pickle.HIGHEST_PROTOCOL)

# TODO: parametrize the following for generating string activation and full tab,
# based on arguments given on top (todo)

# flat tablature dataset
dataset = data_utils.GuitarTabDataset(task='flat_tablature', 
                                     output_representation='flat_tablature',
                                     history=4)

rep_folder = 'data/track_representation_withRandomness_parts'
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

[x_train, y_train, x_valid, y_valid, x_test, y_test] = dataset.load_data()

flat_tablature_rand_dataset = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid,
    'x_test': x_test,
    'y_test': y_test,
}

with open('data' + os.sep + 'flat_tablature_rand_dataset.pickle', 'wb') as handle:
    pickle.dump(flat_tablature_rand_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
