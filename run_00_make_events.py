# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:20:47 2021

@author: user
"""

import os
import guitarpro as gp
import data_utils
import numpy as np

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

# main_folder = 'data/gp_token_examples'
main_folder = 'data/DadaGP-v1.1/DadaGP-v1.1'

level_A_folders = os.listdir(main_folder)

# pieces_events = []

excepted_pieces = []

max_pitch = -1
min_pitch = 1000

for level_A_folder in level_A_folders:
    level_A_path = os.path.join(main_folder, level_A_folder)
    if os.path.isdir( level_A_path ):
        pieces_events = []
        level_B_folders = os.listdir( level_A_path )
        for level_B_folder in level_B_folders:
            level_B_path = os.path.join( level_A_path, level_B_folder )
            for file in os.listdir( level_B_path ):
                if file[-4:-1] == '.gp':
                    print( os.path.join(level_B_path, file) )
                    excepted = False
                    try:
                        gpPieceEvent = data_utils.GPPieceEvents( os.path.join(level_B_path, file) )
                    except:
                        excepted = True
                        print('EXCEPTED')
                        excepted_pieces.append( os.path.join(level_B_path, file) )
                    if len(gpPieceEvent.track_events) > 0 and not excepted:
                        pieces_events.append( gpPieceEvent )
                        if gpPieceEvent.max_pitch > max_pitch:
                            max_pitch = gpPieceEvent.max_pitch
                        if  gpPieceEvent.min_pitch < min_pitch:
                            min_pitch = gpPieceEvent.min_pitch
        with open('data/dadaGP_event_parts/part_'+level_A_folder+'_track_events.pickle', 'wb') as handle:
            pickle.dump(pieces_events, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'excepted_pieces.pickle', 'wb') as handle:
    pickle.dump(excepted_pieces, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 
'''
with open('data' + os.sep + 'track_events.pickle', 'wb') as handle:
    pickle.dump(pieces_events, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data' + os.sep + 'track_events.pickle', 'rb') as handle:
    b = pickle.load(handle)
'''
# %% 

# check time GCD
onset_times = []
durations = []
for p in pieces_events:
    for t in p.track_events:
        tmp_onsets = []
        tmp_durations = []
        for e in t:
            tmp_onsets.append( e['onset_piece'] )
            tmp_durations.append( e['duration'] )
        onset_times.append( tmp_onsets )
        durations.append( tmp_durations)

'''
for ts in onset_times:
    for t in ts:
        if np.floor(t/80) != t/80:
            print(t)

'''
m = 1000000
gcds = []
for ts in onset_times:
    if len(ts) > 0:
        o = np.array(ts)
        gcds.append( np.gcd.reduce(o) )
        d = np.diff(o)
        dnz = d[d!=0]
        if m > np.min(dnz):
            m = np.min(dnz)