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