# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:20:47 2021

@author: user
"""

import os
import guitarpro as gp
# from gp2events import GPPieceEvents
import gp2events
import numpy as np

import pickle

# %% 

main_folder = 'TextoGP/gp_token_examples'

level_A_folders = os.listdir(main_folder)

pieces_events = []

for level_A_folder in level_A_folders:
    level_A_path = os.path.join(main_folder, level_A_folder)
    if os.path.isdir( level_A_path ):
        level_B_folders = os.listdir( level_A_path )
        for level_B_folder in level_B_folders:
            level_B_path = os.path.join( level_A_path, level_B_folder )
            for file in os.listdir( level_B_path ):
                if file[-4:-1] == '.gp':
                    print( os.path.join(level_B_path, file) )
                    gpPieceEvent = gp2events.GPPieceEvents( os.path.join(level_B_path, file) )
                    if len(gpPieceEvent.track_events) > 0:
                        pieces_events.append( gpPieceEvent )

# %% 

with open('filename.pickle', 'wb') as handle:
    pickle.dump(pieces_events, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)

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