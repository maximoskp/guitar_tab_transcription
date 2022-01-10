from my_midi_tools import my_piano_roll, onsetEvents2tabreadyEvents
from my_midi_tools import my_chordify, my_read_midi_mido, tabEvents2gp5
import os
import sys
sys.path.insert(1, '..')
import data_utils
import matplotlib.pyplot as plt
import mido

# To run model for pieces go to
# flat_tab_in_CNN and run make_figs_midi_tab_flat_CNN.py

# %%

# folder = 'midifiles'
folder = '../data/guitar_midi_files/testfiles'
pieces = os.listdir(folder)

# %% run example

idx = 2
m, ticks_per_beat, metadata = my_read_midi_mido( os.path.join(folder, pieces[idx]) )
mid = mido.MidiFile( os.path.join(folder, pieces[idx]) )
duration_events, onset_events = my_chordify(m)

# %%

tabReadyEvents = onsetEvents2tabreadyEvents(onset_events, parts_per_quarter=ticks_per_beat)

# from the following, keep the pianoroll_changes
trep = data_utils.TrackRepresentation(tabReadyEvents)

song = tabEvents2gp5(tabReadyEvents, ticks_per_beat, metadata)

# %% 

plt.clf()
plt.imshow(trep.pianoroll_changes, cmap='gray_r')