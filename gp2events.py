# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:32:32 2021

@author: user
"""

import guitarpro as gp
import os

class GPPieceEvents:
    def __init__(self, file_path):
        song = gp.parse( file_path )
        self.name = file_path.split( os.sep )[-1]
        self.track_events = []
        tracks = song.tracks
        aborted = False
        for track in tracks:
            strings = track.strings
            # check if proper guitar tunning
            proper_guitar = True
            proper_tunning = [64, 59, 55, 50, 45, 40] # make static
            for i, s in enumerate(strings):
                if i >= len(proper_tunning) or s.value != proper_tunning[i]:
                    print( file_path + ' - ' + str(s.value) + ': tunning not proper - ABORTING')
                    proper_guitar = False
                    aborted = True
                    break
            if proper_guitar:
                measures = track.measures
                note_events = []
                for measure in measures:
                    voices = measure.voices
                    for voice in voices:
                        beats = voice.beats
                        for beat in beats:
                            if beat.effect.mixTableChange:
                                print( file_path + ': mixTableChange - ABORTING beat')
                                # aborted = True
                            else:
                                if beat.status.name != 'normal':
                                    print( file_path + ': not normal - ABORTING beat')
                                    # aborted = True
                                else:
                                    note_event = {}
                                    note_event['duration'] = beat.duration.time
                                    note_event['onset_piece'] = beat.start
                                    note_event['onset_measure'] = beat.startInMeasure
                                    note_event['pitches'] = []
                                    # only normal notes appended
                                    for n in beat.notes:
                                        if n.type.value == 1:
                                            pitch_event = {}
                                            pitch_event['string'] = n.string
                                            pitch_event['fret'] = n.value
                                            pitch_event['pitch'] = n.realValue
                                            pitch_event['velocity'] = n.velocity
                                            pitch_event['duration_percentage'] = n.durationPercent
                                            note_event['pitches'].append( pitch_event )
                                            note_events.append( note_event )
                                        else:
                                            print(file_path + 'note type NOT 1 - ABORTING event')
            if not aborted:
                self.track_events.append(note_events)