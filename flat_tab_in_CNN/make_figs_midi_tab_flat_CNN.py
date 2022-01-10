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
from my_midi_tools import my_chordify, my_read_midi_mido, tabEvents2gp5
import guitarpro as gp

# load model
model = keras.models.load_model( '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_current_best.hdf5' )

# folder = 'midifiles'
folder = '../data/guitar_midi_files/testfiles'
pieces = os.listdir(folder)

idx = 2
# TODO: get tempo when reading midi
# TODO: get time signature
m, ticks_per_beat, metadata = my_read_midi_mido( os.path.join(folder, pieces[idx]) )
duration_events, onset_events = my_chordify(m)

tabReadyEvents = onsetEvents2tabreadyEvents(onset_events, parts_per_quarter=ticks_per_beat)

# from the following, keep the pianoroll_changes - Don't need it
# trep = data_utils.TrackRepresentation(tabReadyEvents, keep_events=True)

tmp_tab = np.zeros( 150*4 ) # tab size * history
midi_fretboard = m2t.get_midi_full_fretboard()

print('making new tab')
for i, ev in enumerate(tabReadyEvents):
    print(str(i) + '/' + str(len(tabReadyEvents)))
    event_pitches = [p['pitch'] for p in ev['pitches']]
    # put within fretboard range
    for ii in range( len( event_pitches ) ):
        while event_pitches[ii] < np.min(midi_fretboard):
            event_pitches[ii] += 12
            ev['pitches'][ii]['pitch'] = event_pitches[ii]
        while event_pitches[ii] > np.max(midi_fretboard):
            event_pitches[ii] -= 12
            ev['pitches'][ii]['pitch'] = event_pitches[ii]
    # print('event_pitches: ', event_pitches)
    midi_frame = np.zeros( 128 ).astype(int)
    midi_frame[ event_pitches ] = 1
    tmp_in = np.reshape( np.r_[ midi_frame , tmp_tab ] , [1,728] )
    y_pred = model.predict( [ tmp_in ] )
    decision = m2t.midi_and_flat_tab_decision( midi_frame, y_pred[0] )
    dstring = np.where( decision != 0 )[0]
    dfret = np.where( decision != 0 )[1]
    for i in range( len(dstring) ):
        tmp_midi_pitch = midi_fretboard[ dstring[i] , dfret[i] ]
        # ev_idx = event_pitches.index( tmp_midi_pitch )
        ev_idxs = [i for i, x in enumerate(event_pitches) if x == tmp_midi_pitch]
        for ev_idx in ev_idxs:
            ev['pitches'][ev_idx]['string'] = dstring[i]
            ev['pitches'][ev_idx]['fret'] = dfret[i]
    tmp_tab = np.roll( tmp_tab , 150 )
    tmp_tab[:150] = np.reshape(decision, 150 )

# write gp5
s = tabEvents2gp5( tabReadyEvents, parts_per_quarter=ticks_per_beat, metadata=metadata )
gp.write(s, 'gp_files/' + pieces[idx].split('.')[0] + '.gp4' )
