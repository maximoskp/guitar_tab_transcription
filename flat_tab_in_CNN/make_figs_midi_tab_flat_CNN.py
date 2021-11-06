#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 06:49:00 2021

@author: max
"""

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
import sys
sys.path.insert(1, '..')
import data_utils
sys.path.insert(1, '../run_for_midi_files')
from my_midi_tools import my_piano_roll, onsetEvents2tabreadyEvents
from my_midi_tools import my_chordify, my_read_midi_mido

# %%

# load model
model = keras.models.load_model( '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_current_best.hdf5' )

# %% load midi file

# folder = 'midifiles'
folder = '../data/guitar_midi_files/testfiles'
pieces = os.listdir(folder)

idx = 0
m, ticks_per_beat = my_read_midi_mido( os.path.join(folder, pieces[idx]) )
duration_events, onset_events = my_chordify(m)

tabReadyEvents = onsetEvents2tabreadyEvents(onset_events, parts_per_quarter=ticks_per_beat)

# from the following, keep the pianoroll_changes
trep = data_utils.TrackRepresentation(tabReadyEvents)

# %% 

tmp_tab = np.zeros( 150*4 ) # tab size * history
midi_frames = trep.pianoroll_changes

for i in range(midi_frames.shape[1]):
    tmp_in = np.reshape( np.r_[ midi_frames[:,i] , tmp_tab ] , [1,728] )
    y_pred = model.predict( [ tmp_in ] )
    decision = m2t.midi_and_flat_tab_decision( midi_frames[:,i], y_pred[0] )
    dstring = np.where( decision != 0 )[0]
    dfret = np.where( decision != 0 )[1]
    trep.tablature_changes[dstring,i] = dfret
    tmp_tab = np.roll( tmp_tab , 150 )
    tmp_tab[:150] = np.reshape(decision, 150 )