# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:32:32 2021

@author: user
"""

import guitarpro as gp
import os
import numpy as np
import matplotlib.pyplot as plt

class GPPieceEvents:
    def __init__(self, file_path):
        song = gp.parse( file_path )
        self.name = file_path.split( os.sep )[-1]
        self.track_events = []
        tracks = song.tracks
        aborted = False
        self.max_pitch = -1
        self.min_pitch = 1000
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
                                            if pitch_event['pitch'] > self.max_pitch:
                                                self.max_pitch = pitch_event['pitch']
                                            if pitch_event['pitch'] < self.min_pitch:
                                                self.min_pitch = pitch_event['pitch']
                                            pitch_event['velocity'] = n.velocity
                                            pitch_event['duration_percentage'] = n.durationPercent
                                            note_event['pitches'].append( pitch_event )
                                            note_events.append( note_event )
                                        else:
                                            print(file_path + 'note type NOT 1 - ABORTING event')
            if not aborted:
                if len( note_events ) > 0:
                    self.track_events.append(note_events)
# end class GPPieceEvents

class TrackRepresentation():
    def __init__(self, track, piece_name='undefined', track_number=-1):
        self.piece_name = piece_name
        self.track_number = track_number
        onsets = np.array( [ t['onset_piece'] for t in track ] )
        onsets -= onsets[0]
        g = np.gcd.reduce(onsets)
        onsets = (onsets/g).astype('int')
        
        durations = np.array( [ t['duration'] for t in track ] )
        durations = np.floor( durations/g ).astype( 'int' )
        durations[durations==0] = 1
        
        self.pianoroll = np.zeros( ( 128 , onsets[-1]+durations[-1] ) )
        self.onsetsroll = np.zeros( ( 128 , onsets[-1]+durations[-1] ) )
        
        for i, t in enumerate(track):
            pitches = t['pitches']
            for p in pitches:
                tmp_duration = np.max( [np.floor( durations[i]/p['duration_percentage'] ), 1])
                tmp_velocity = p['velocity']
                for d in range(tmp_duration.astype('int')):
                    if d == 0:
                        self.onsetsroll[ p['pitch'] , onsets[i]+d ] = tmp_velocity
                    self.pianoroll[ p['pitch'] , onsets[i]+d ] = tmp_velocity
        
        # keep only active range of notes
        # self.pianoroll = self.pianoroll[40:95, :]
        # self.onsetsroll = self.onsetsroll[40:95, :]
        
        self.tablature = -1*np.ones( ( 6 , onsets[-1]+durations[-1] ) )
        self.string_activation = np.zeros( ( 6 , onsets[-1]+durations[-1] ) )
        
        for i, t in enumerate(track):
            pitches = t['pitches']
            for p in pitches:
                self.tablature[ p['string']-1 , onsets[i] ] = p['fret']
                self.string_activation[ p['string']-1 , onsets[i] ] = 1
        
        # remove zeros
        nz_idxs = np.sum(self.pianoroll, axis=0)!=0
        p0 = self.pianoroll[:, nz_idxs]
        # get difference idxs
        d = np.diff(p0, axis=1)
        dsum = np.sum( np.abs(d), axis=0)
        idx2keep = np.append(0, np.where( dsum != 0 )[0] + 1 )
        
        self.pianoroll_changes = p0[:, idx2keep]
        t0 = self.tablature[:, nz_idxs]
        self.tablature_changes = t0[:, idx2keep]
        s0 = self.string_activation[:, nz_idxs]
        self.string_activation_changes = s0[:, idx2keep]
    # end constructor
    
    def plot_pianoroll_part(self, start_idx=0, end_idx=50):
        plt.imshow( self.pianoroll[:,start_idx:end_idx], cmap='gray_r', origin='lower' )
    # end plot_pianoroll_part
    
    def plot_tab_part(self, start_idx=0, end_idx=50):
        tablature_part = self.tablature[:,start_idx:end_idx]
        x = np.arange(tablature_part.shape[1])
        x_length = len(x)
        y_height = x_length/5.
        
        y_offset = y_height/10.
        y_room = y_height - 2*y_offset
        
        plt.clf()
        for string in range(6):
            # plot string
            string_height = y_offset + (6-string)*y_room/6
            plt.plot( [x[0], x[-1]] , [string_height ,string_height], 'gray' )
            for i, f in enumerate(tablature_part[string,:]):
                if f > -1:
                    plt.text(i, string_height, str(f.astype(int)))
        plt.axis('equal')
    # end plot_tab_part