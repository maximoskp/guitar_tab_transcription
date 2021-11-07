# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 07:29:30 2021

@author: user
"""

import os
import guitarpro as gp

# folder = 'TextoGP/gp_token_examples/A/Aguado, Dionso'
# file = 'Aguado, Dionso - Walzer.gp4'
folder = 'data/testGPfiles'
file = 'Abba - Gimme! Gimme! Gimme!.gp4'

# %% 

song = gp.parse( os.path.join( folder, file ) )

tracks = song.tracks
t0 = tracks[0]

# check if guitar with proper tunning
strings = t0.strings
proper_guitar = True
proper_tunning = [64, 59, 55, 50, 45, 40]
for i, s in enumerate(strings):
    if s.value != proper_tunning[i]:
        print(str(s.value) + ' - tunning not proper')
if proper_guitar:
    print('proper tunning')

measures = t0.measures
m0 = measures[0]

voices = m0.voices
v0 = voices[0]

beats = v0.beats
b0 = beats[0]

# check if instruments changes and discard track if so
if b0.effect.mixTableChange:
    print('mixTableChange')
else:
    print('NO mixTableChange')

# only normal notes
if b0.status.name == 'normal':
    note_event = {}
    note_event['duration'] = b0.duration.time
    note_event['onset_piece'] = b0.start
    note_event['onset_measure'] = b0.startInMeasure
    note_event['pitches'] = []
    # only normal notes appended
    for n in b0.notes:
        if n.type.value == 1:
            pitch_event = {}
            pitch_event['string'] = n.string
            pitch_event['fret'] = n.value
            pitch_event['pitch'] = n.realValue
            pitch_event['velocity'] = n.velocity
            pitch_event['duration_percentage'] = n.durationPercent
            note_event['pitches'].append( pitch_event )
        else:
            print('note type NOT 1')
else:
    print('note NOT normal')