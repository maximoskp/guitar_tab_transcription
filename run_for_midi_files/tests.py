from my_midi_tools import my_piano_roll, onsetEvents2tabreadyEvents
from my_midi_tools import my_chordify, my_read_midi_mido
import os
import sys
sys.path.insert(1, '..')
import data_utils

# %%

# folder = 'midifiles'
folder = '../data/guitar_midi_files/testfiles'
pieces = os.listdir(folder)

# %% run example

idx = 0
m, ticks_per_beat = my_read_midi_mido( os.path.join(folder, pieces[idx]) )
duration_events, onset_events = my_chordify(m)

# %%

tabReadyEvents = onsetEvents2tabreadyEvents(onset_events, parts_per_quarter=ticks_per_beat)

# from the following, keep the pianoroll_changes
trep = data_utils.TrackRepresentation(tabReadyEvents)