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

# binary tab -----------------------------------------------------------------
def tabFrame2binary(t):
    b = np.zeros(5*6)
    for i in range(6):
        s = np.binary_repr( t[i].astype(int) , width=5 )
        b[i*5:i*5+5] = np.array(list(s), dtype=np.float32)
    return b

def tablature2binary(tablature):
    b = np.zeros( (5*6 , tablature.shape[1]) )
    for i in range(tablature.shape[1]):
        b[:, i] = tabFrame2binary(tablature[:, i])
    return b

def bool2int(x):
    y = 0
    for i,j in enumerate(np.flip(x)):
        y += j<<i
    return y

def binary2tablature(b):
    t = np.zeros( (6, b.shape[1]) )
    for i in range(b.shape[1]):
        for j in range(6):
            t[j,i] = bool2int( b[j*5:j*5+5, i].astype(int) )
    t[ t==31 ] = -1
    return t

# flat tab -------------------------------------------------------------------

def tabFrame2flatFretboard(t, frets_num=24):
    f = np.zeros( (6, frets_num+1) ) # 0-th fret is counted
    for i in range(len(t)):
        if t[i] >= 0 and t[i] <= frets_num:
            f[i, t[i].astype(int)] = 1
    # print('before flat: ', f.shape)
    # print('after flat: ', f.flatten().shape)
    return f.flatten()

def tablature2flatFretboard(tablature, frets_num=24):
    f = np.zeros( ( 6*(frets_num+1) , tablature.shape[1] ) )
    for i in range( tablature.shape[1] ):
        f[:,i] = tabFrame2flatFretboard( tablature[:, i], frets_num=frets_num )
    return f

# full tab 3D ----------------------------------------------------------------
def tabFrame2Fretboard(t, frets_num=24):
    f = np.zeros( (6, frets_num+1) ) # 0-th fret is counted
    for i in range(len(t)):
        if t[i] >= 0 and t[i] <= frets_num:
            f[i, t[i].astype(int)] = 1
    # print('before flat: ', f.shape)
    # print('after flat: ', f.flatten().shape)
    return f

# pianoroll column 2 tokens -------------------------------------------------
def pianorollFrame2tokens(c, compount=True):
    nz = np.where( c != 0 )[0]
    s = []
    for i in nz:
        if compount:
            s.append( 'note_' + str( i ) )
        else:
            s.append( 'note' )
            s.append( str( i ) )
    return s

# pianoroll 2 tokens ---------------------------------------------------------
def pianoroll2tokens(p, compount=True):
    s = ['<SOS>']
    for i in range( p.shape[1] ):
        c = pianorollFrame2tokens( p[:,i], compount )
        if len( c ) > 0:
            s.append( 'new_frame' )
            s.extend( c )
    s.append('<EOS>')
    return s

# tablature string 2 tokens ---------------------------------------------------------
string_names = ['E', 'A', 'D', 'G', 'B', 'E']
def tablatureFrame2tokens(c, compount=True):
    nz = np.where( c != -1 )[0]
    s = []
    for i in nz:
        if compount:
            s.append('string_' + string_names[ i ])
            s.append('fret_' + str(c[i].astype(int)))
        else:
            s.append('string')
            s.append( string_names[ i ] )
            s.append('fret')
            s.append(str(c[i].astype(int)))
    return s

# tablature 2 tokens ---------------------------------------------------------
def tablature2tokens(p, compount=True):
    s = ['<SOS>']
    for i in range( p.shape[1] ):
        c = tablatureFrame2tokens( p[:,i], compount )
        if len( c ) > 0:
            s.append( 'new_frame' )
            s.extend( c )
    s.append('<EOS>')
    return s

# %% plottings
def plot_full_tabs(t, titles=None):
    for i in range(t.shape[0]):
        plt.subplot( 4, (t.shape[0]-1)//4+1, i+1 )
        plt.imshow(t[i,:,:], cmap='gray_r')
        if titles is not None:
            if len(titles) == t.shape[0]:
                plt.title( titles[i] )
            else:
                if i == t.shape[0]-1:
                    plt.title(titles)
        else:
            plt.title(str( i ))

def plot_flat_tabs(t, titles=None):
    for i in range(t.shape[1]):
        plt.subplot( 4, (t.shape[1]-1)//4+1, i+1 )
        plt.imshow( np.reshape( t[:,i], [6,25] ), cmap='gray_r')
        if titles is not None:
            if len(titles) == t.shape[1]:
                plt.title( titles[i] )
            else:
                if i == t.shape[1]-1:
                    plt.title(titles)
        else:
            plt.title(str( i ))

def tablature2Fretboard(tablature, frets_num=24):
    f = np.zeros( ( tablature.shape[1] , 6, frets_num+1 ) )
    for i in range( tablature.shape[1] ):
        f[i,:,:] = tabFrame2Fretboard( tablature[:, i], frets_num=frets_num )
    return f

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
                    # print( file_path + ' - ' + str(s.value) + ': tunning not proper - ABORTING')
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
                                pass
                                # print( file_path + ': mixTableChange - ABORTING beat')
                                # aborted = True
                            else:
                                if beat.status.name != 'normal':
                                    pass
                                    # print( file_path + ': not normal - ABORTING beat')
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
                                            pass
                                            # print(file_path + 'note type NOT 1 - ABORTING event')
            if not aborted:
                if len( note_events ) > 0:
                    self.track_events.append(note_events)
# end class GPPieceEvents

class TrackRepresentation():
    def __init__(self, track, piece_name='undefined', track_number=-1, keep_full=False, keep_events=False, random_pr=None):
        self.piece_name = piece_name
        self.track_number = track_number
        self.keep_full= keep_full
        self.keep_events= keep_events
        if self.keep_events:
            self.events = track
        onsets = np.array( [ t['onset_piece'] for t in track ] )
        onsets -= onsets[0]
        g = np.gcd.reduce(onsets)
        if g > 0:
            onsets = (onsets/g).astype('int')
        else:
            onsets = onsets.astype('int')
        
        durations = np.array( [ t['duration'] for t in track ] )
        if g > 0:
            durations = np.floor( durations/g ).astype( 'int' )
        else:
            durations = np.floor( durations ).astype('int')
        durations[durations==0] = 1
        
        self.pianoroll = np.zeros( ( 128 , onsets[-1]+durations[-1] ), dtype=np.float32)
        self.onsetsroll = np.zeros( ( 128 , onsets[-1]+durations[-1] ), dtype=np.float32 )
        
        for i, t in enumerate(track):
            pitches = t['pitches']
            for p in pitches:
                tmp_duration = np.max( [np.floor( durations[i]/p['duration_percentage'] ), 1])
                tmp_velocity = p['velocity']
                for d in range(tmp_duration.astype('int')):
                    # check if random components need to be added in the pianoroll
                    random_pitch = -1
                    if random_pr is not None:
                        if np.random.rand() <= random_pr:
                            random_pitch = p['pitch'] + [-12, -5, -4, -3, 3, 4, 7, 12][np.random.randint(8)]
                    if d == 0:
                        self.onsetsroll[ p['pitch'] , onsets[i]+d ] = tmp_velocity
                        if random_pitch >= 0:
                            self.onsetsroll[ random_pitch , onsets[i]+d ] = tmp_velocity
                    self.pianoroll[ p['pitch'] , onsets[i]+d ] = tmp_velocity
                    if random_pitch >= 0:
                        self.pianoroll[ random_pitch , onsets[i]+d ] = tmp_velocity
        
        # keep only active range of notes
        # self.pianoroll = self.pianoroll[40:95, :]
        # self.onsetsroll = self.onsetsroll[40:95, :]
        
        self.tablature = -1*np.ones( ( 6 , onsets[-1]+durations[-1] ), dtype=np.float32 )
        self.string_activation = np.zeros( ( 6 , onsets[-1]+durations[-1] ), dtype=np.float32 )
        
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
        if self.keep_events:
            tmp_all_idxs = np.arange( onsets[-1]+durations[-1] ).astype(int)
            tmp_nz_idxs = tmp_all_idxs[nz_idxs]
            self.event_onsets_kept = tmp_nz_idxs[ np.array(idx2keep, dtype=int) ]*g
        
        self.pianoroll_changes = p0[:, idx2keep]
        t0 = self.tablature[:, nz_idxs]
        self.tablature_changes = t0[:, idx2keep]
        s0 = self.string_activation[:, nz_idxs]
        self.string_activation_changes = s0[:, idx2keep]
        if not self.keep_full:
            del self.pianoroll
            del self.onsetsroll
            del self.tablature
            del self.string_activation
    # end constructor
    
    def plot_pianoroll_part(self, start_idx=0, end_idx=50):
        plt.imshow( self.pianoroll_changes[:,start_idx:end_idx], cmap='gray_r', origin='lower' )
    # end plot_pianoroll_part
    
    def plot_tab_part(self, start_idx=0, end_idx=50):
        tablature_part = self.tablature_changes[:,start_idx:end_idx]
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

    def tab2events(self):
        if not self.keep_events:
            print('ERROR: events should have been kept')
            return
        # start reading midi and tab changes
        i = 0
        for ev in self.events:
            # get pitches of event
            p = [ n['pitch'] for n in ev.pitches ]

    # end tab2events
# end TrackRepresentation

class GuitarTabDataset():
    def __init__(self, history=2, task='string_activation',
                 output_representation='binary_tab',):
        self.history = history
        self.task = task
        self.output_representation = output_representation
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
    def add_matrices_old(self, r):
        # add from TrackRepresentation object
        tmp_all_x = np.concatenate( (np.zeros((r.pianoroll_changes.shape[0], self.history)), r.pianoroll_changes ), axis=1)
        tmp_x = tmp_all_x[:, self.history:]
        for i in range(1, self.history+1, 1):
            tmp_x = np.vstack( (tmp_x , tmp_all_x[:, self.history-i:-i]) )
        self.pianoroll_changes.append( tmp_x )
        if self.output_representation == 'binary_tab':
            self.tablature_changes.append( tablature2binary(r.tablature_changes) )
        elif self.output_representation == 'flat_tablature':
            self.tablature_changes.append( tablature2flatFretboard(r.tablature_changes) )
        elif self.output_representation == 'full_tablature':
            self.tablature_changes.append( tablature2Fretboard(r.tablature_changes) )
        else:
            print('unknown output_representation')
        self.string_activation_changes.append( r.string_activation_changes )
        # self.tablature_changes.append( np.concatenate( (np.zeros((r.tablature_changes.shape[0], self.history)), r.tablature_changes ), axis=1) )
        # self.string_activation_changes.append( np.concatenate( (np.zeros((r.string_activation_changes.shape[0], self.history)), r.string_activation_changes ), axis=1) )
    # end add_matrices_old
    def add_matrices(self, r):
        # add from TrackRepresentation object
        # tmp_all_x = np.concatenate( (np.zeros((r.pianoroll_changes.shape[0], self.history)), r.pianoroll_changes ), axis=1)
        if self.output_representation == 'binary_tab':
            # TODO: put binary tab history
            tmp_all_x = np.concatenate( (np.zeros((r.pianoroll_changes.shape[0], self.history)), r.pianoroll_changes ), axis=1)
            tmp_x = tmp_all_x[:, self.history:]
        elif self.output_representation == 'flat_tablature':
            tmp_flat_tab = tablature2flatFretboard(r.tablature_changes)
            tmp_all_x = np.concatenate( (np.zeros((tmp_flat_tab.shape[0], self.history)), tmp_flat_tab ), axis=1)
            tmp_x = r.pianoroll_changes
        if self.output_representation == 'flat_tablature' or self.output_representation == 'binary_tab':
            for i in range(1, self.history+1, 1):
                tmp_x = np.vstack( (tmp_x , tmp_all_x[:, self.history-i:-i]) )
            self.pianoroll_changes.append( tmp_x.astype(bool) )
        if self.output_representation == 'binary_tab':
            self.tablature_changes.append( tablature2binary(r.tablature_changes) )
        elif self.output_representation == 'flat_tablature':
            self.tablature_changes.append( tablature2flatFretboard(r.tablature_changes).astype(bool) )
        elif self.output_representation == 'full_tablature':
            self.tablature_changes.append( tablature2Fretboard(r.tablature_changes) )
        else:
            print('unknown output_representation')
        # self.string_activation_changes.append( r.string_activation_changes )
        # self.tablature_changes.append( np.concatenate( (np.zeros((r.tablature_changes.shape[0], self.history)), r.tablature_changes ), axis=1) )
        # self.string_activation_changes.append( np.concatenate( (np.zeros((r.string_activation_changes.shape[0], self.history)), r.string_activation_changes ), axis=1) )
    # end add_matrices
    def load_data(self, train_ratio=0.8, validation=True, validation_ratio=0.2):
        self.validation = validation
        # shuffled_idxs = np.arange( len( self.pianoroll_changes ) )
        # np.random.shuffle( shuffled_idxs )
        # self.pianoroll_changes, self.tablature_changes, self.string_activation_changes = shuffle( self.pianoroll_changes,
        #                                          self.tablature_changes,
        #                                          self.string_activation_changes)
        self.pianoroll_changes, self.tablature_changes= shuffle( self.pianoroll_changes,
                                                 self.tablature_changes)
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
    def load_full_tabs(self, train_ratio=0.8, validation=True, validation_ratio=0.2):
        self.validation = validation
        self.tablature_changes = shuffle( self.tablature_changes )
        train_idx = np.floor( len( self.tablature_changes )*train_ratio ).astype(int)
        valid_idx = 0
        if self.validation:
            valid_idx = np.floor( train_idx*validation_ratio ).astype(int)
            x_valid = self.tablature_changes[train_idx-valid_idx:train_idx]
        x_train = self.tablature_changes[:train_idx-valid_idx]
        x_test = self.tablature_changes[train_idx:]
        
        self.x_train = np.concatenate( x_train , axis=0 )
        self.x_test = np.concatenate( x_test , axis=0 )
        
        if self.validation:
            self.x_valid = np.concatenate( x_valid , axis=0 )
        
        if self.validation:
            return [self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test]
        else:
            return [self.x_train, self.y_train, self.x_test, self.y_test]
    # end load_full_tabs
# end GuitarTabDataset