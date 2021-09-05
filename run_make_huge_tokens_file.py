# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 08:33:56 2021

@author: user
"""

import pickle
import gp2events
import os

# %% 

pianoroll_tokens_u = []

rep_folder = 'data/track_representation_parts'
reps = os.listdir( rep_folder )

for f_name in reps:
    f_path = os.path.join( rep_folder, f_name )
    b = None
    print(f_path)
    with open(f_path, 'rb') as handle:
        b = pickle.load(handle)
    for r in b:
        pianoroll_tokens_u.append( gp2events.pianoroll2tokens( r.pianoroll_changes ) )

with open('data' + os.sep + 'pianoroll_tokens_u.pickle', 'wb') as handle:
    pickle.dump(pianoroll_tokens_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

del pianoroll_tokens_u

# 

tablature_tokens_u = []

rep_folder = 'data/track_representation_parts'
reps = os.listdir( rep_folder )

for f_name in reps:
    f_path = os.path.join( rep_folder, f_name )
    b = None
    print(f_path)
    with open(f_path, 'rb') as handle:
        b = pickle.load(handle)
    for r in b:
        tablature_tokens_u.append( gp2events.tablature2tokens( r.tablature_changes ) )


with open('data' + os.sep + 'tablature_tokens_u.pickle', 'wb') as handle:
    pickle.dump(tablature_tokens_u, handle, protocol=pickle.HIGHEST_PROTOCOL)
