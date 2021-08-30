# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:32:32 2021

@author: user
"""

import guitarpro as gp
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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

class GuitarTabDataset():
    def __init__(self, history=2):
        self.history = history
        # collections of matrices
        self.pianoroll_changes = []
        self.tablature_changes = []
        self.string_activation_changes = []
        # final matrices
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
    # end constructor
    def add_matrices(self, r):
        # add from TrackRepresentation object
        tmp_all_x = np.concatenate( (np.zeros((r.pianoroll_changes.shape[0], self.history)), r.pianoroll_changes ), axis=1)
        tmp_x = tmp_all_x[:, self.history:]
        for i in range(1, self.history+1, 1):
            tmp_x = np.vstack( (tmp_x , tmp_all_x[:, self.history-i:-i]) )
        self.pianoroll_changes.append( tmp_x )
        self.tablature_changes.append( r.tablature_changes )
        self.string_activation_changes.append( r.string_activation_changes )
        # self.tablature_changes.append( np.concatenate( (np.zeros((r.tablature_changes.shape[0], self.history)), r.tablature_changes ), axis=1) )
        # self.string_activation_changes.append( np.concatenate( (np.zeros((r.string_activation_changes.shape[0], self.history)), r.string_activation_changes ), axis=1) )
    # end add_matrices
    def load_data(self, task='string_activation', train_ratio=0.8,
                  validation=True, validation_ratio=0.2):
        self.task = task
        self.validation = validation
        # shuffled_idxs = np.arange( len( self.pianoroll_changes ) )
        # np.random.shuffle( shuffled_idxs )
        self.pianoroll_changes, self.tablature_changes, self.string_activation_changes = shuffle( self.pianoroll_changes,
                                                 self.tablature_changes,
                                                 self.string_activation_changes)
        train_idx = np.floor( len( self.pianoroll_changes )*train_ratio ).astype(int)
        valid_idx = 0
        if self.validation:
            valid_idx = np.floor( train_idx*validation_ratio ).astype(int)
            x_valid = self.pianoroll_changes[train_idx-valid_idx:train_idx]
        x_train = self.pianoroll_changes[:train_idx-valid_idx]
        x_test = self.pianoroll_changes[train_idx:]
        
        self.x_train = np.concatenate( x_train , axis=1 )
        self.x_test = np.concatenate( x_test , axis=1 )
        
        if self.validation:
            self.x_valid = np.concatenate( x_valid , axis=1 )
        if self.task == 'string_activation':
            y = self.string_activation_changes
        else:
            y = self.tablature_changes
        if self.validation:
            y_valid = y[train_idx-valid_idx:train_idx]
        # self.y1 = y
        y_train = y[:train_idx-valid_idx]
        y_test = y[train_idx:]
        self.y_train = np.concatenate( y_train , axis=1 )
        self.y_test = np.concatenate( y_test , axis=1 )
        if self.validation:
            self.y_valid = np.concatenate( y_valid , axis=1 )
        if self.validation:
            return [self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test]
        else:
            return [self.x_train, self.y_train, self.x_test, self.y_test]
    # end load_data