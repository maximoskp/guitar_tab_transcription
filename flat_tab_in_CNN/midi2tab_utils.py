# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 21:16:17 2021

@author: user
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt

def get_midi_full_fretboard():
    proper_tunning = np.array( [64, 59, 55, 50, 45, 40] )
    midi_full_fretboard = np.zeros( (6,25) )
    for i in range(25):
        midi_full_fretboard[:,i] = proper_tunning + i
    return midi_full_fretboard
# end get_midi_full_fretboard

def make_all_binary_tabs_for_binary_midi(m):
    midi_notes = np.array( np.where( m )[0] )
    # print('midi_notes: ', midi_notes)
    # keep structures as [pitch, string, fret]
    all_structs = []
    f = get_midi_full_fretboard()
    for i, n in enumerate(midi_notes):
        # in case n is out of fretboard range
        while n < np.min(f):
            n += 12
            # print('n1: ', n)
        while n > np.max(f):
            # print('n2: ', n)
            n -= 12
        tmp_where = np.where( n == f )
        # print('tmp_where: ', tmp_where)
        tmp_strings_of_note = tmp_where[0]
        tmp_frets_of_note = tmp_where[1]
        for j in range(len(tmp_strings_of_note)):
            all_structs.append( [n, tmp_strings_of_note[j], tmp_frets_of_note[j]] )
    print('all_structs: ')
    print(all_structs)
    # make all combinations
    all_combinations = list(itertools.combinations( all_structs, min(6, len(midi_notes)) ))
    print('all_combinations')
    print(all_combinations)
    # keep only combinations with unique pitches and unique strings
    combinations2keep = []
    for c in all_combinations:
        tmp_pitches = []
        tmp_strings = []
        keep_combination = True
        for p in c:
            if p[0] in tmp_pitches or p[1] in tmp_strings:
                keep_combination = False
                break
            else:
                tmp_pitches.append( p[0] )
                tmp_strings.append( p[1] )
        if keep_combination:
            combinations2keep.append( c )
    print('combinations2keep')
    print(combinations2keep)
    # keep structures as [pitch, string, fret]
    all_binary_fretboards = []
    for combination in combinations2keep:
        b = np.zeros( (6,25) )
        # get fret for each string
        # print('c: ', c)
        for c in combination:
            b[ c[1], c[2] ] = 1
        all_binary_fretboards.append( b )
    return all_binary_fretboards
    # return combinations2keep
# end make_all_binary_tabs_for_binary_midi

def make_all_binary_tabs_for_binary_midi_old1(m):
    midi_notes = np.array( np.where( m )[0] )
    # print('midi_notes: ', midi_notes)
    # notes number x 6
    strings_notes_matrix = -1*np.ones( (len(midi_notes), 6) ).astype(int)
    f = get_midi_full_fretboard()
    for i, n in enumerate(midi_notes):
        # in case n is out of fretboard range
        while n < np.min(f):
            n += 12
            # print('n1: ', n)
        while n > np.max(f):
            # print('n2: ', n)
            n -= 12
        tmp_where = np.where( n == f )
        # print('tmp_where: ', tmp_where)
        tmp_strings_of_note = tmp_where[0]
        tmp_frets_of_note = tmp_where[1]
        for j in range(len(tmp_strings_of_note)):
            strings_notes_matrix[ i , tmp_strings_of_note[j] ] = tmp_frets_of_note[j]
    print('strings_notes_matrix: ')
    print(strings_notes_matrix)
    # keep nonnegative content of strings
    strings_content = {}
    active_strings = []
    for i in range(strings_notes_matrix.shape[1]):
        tmp_where = np.where( strings_notes_matrix[:,i] >= 0 )[0]
        if len(tmp_where) > 0:
            strings_content[i] = list( tmp_where )
            active_strings.append( i )
    print('strings_content: ')
    print(strings_content)
    print('active_strings: ')
    print(active_strings)
    v = list( strings_content.values() )
    # print(v)
    all_combinations = list( itertools.product( *v ) )
    # print(len(all_combinations))
    print(all_combinations)
    # keep only combinations that include single instances of notes
    combinations2keep = []
    if len(all_combinations) == 1:
        for c in all_combinations[0]:
            combinations2keep.append( [c] )
    else:
        for c in all_combinations:
            if len( np.unique(c) ) == len( c ):
                combinations2keep.append(c)
    # print(len(combinations2keep))
    # print(combinations2keep)
    all_binary_fretboards = []
    for i_comb, c in enumerate(combinations2keep):
        b = np.zeros( (6,25) )
        # get fret for each string
        # print('c: ', c)
        for i,row_idx in enumerate(c):
            if len(all_combinations) == 1:
                string = active_strings[i_comb]
            else:
                string = active_strings[i]
            fret = strings_notes_matrix[row_idx, string]
            # print('i: ', i)
            # print('row_idx: ', row_idx)
            # print('string: ', string)
            # print('fret: ', fret)
            b[ string, fret ] = 1
        all_binary_fretboards.append( b )
    return all_binary_fretboards
# end make_all_binary_tabs_for_binary_midi


def make_all_binary_tabs_for_binary_midi_old(m):
    midi_notes = np.where( m )[0]
    print('midi_notes: ', midi_notes)
    strings_for_notes = []
    frets_for_notes = []
    f = get_midi_full_fretboard()
    for n in midi_notes:
        # in case n is out of fretboard range
        while n < np.min(f):
            n += 12
            print('n1: ', n)
        while n > np.max(f):
            print('n2: ', n)
            n -= 12
        tmp_where = np.where( n == f )
        print('tmp_where: ', tmp_where)
        strings_for_notes.append( tmp_where[0] )
        frets_for_notes.append( tmp_where[1] )
    # get all combinations of strings
    print('====================================')
    print('====================================')
    print('====================================')
    print('strings_for_notes: ', strings_for_notes)
    print('frets_for_notes: ', frets_for_notes)
    print('====================================')
    print('====================================')
    print('====================================')
    all_combinations = list( itertools.product( *strings_for_notes ) )
    print('len(all_combinations): ', len(all_combinations))
    # keep only combinations with different elements
    combinations2keep = []
    for c in all_combinations:
        # check for dublicates
        s = list(set( c ))
        if s not in combinations2keep:
            combinations2keep.append( s )
    # construct binary fretboard for each string combination
    print('len(combinations2keep): ', len(combinations2keep))
    print('combinations2keep: ', combinations2keep)
    all_binary_fretboards = []
    for c in combinations2keep:
        b = np.zeros( (6,25) )
        # get fret for each string
        print('c: ', c)
        for i,string in enumerate(c):
            print('i: ', i)
            print('string: ', string)
            print('strings_for_notes[i]: ', strings_for_notes[i])
            fret_idx = np.where(string == strings_for_notes[i])[0][0].astype(int)
            fret = frets_for_notes[i][fret_idx]
            b[ string, fret ] = 1
        all_binary_fretboards.append( b )
    return all_binary_fretboards
# end make_all_binary_tabs_for_binary_midi

def midi_and_flat_tab_decision(m, t):
    # m: 128x1: midi array
    # t: 150x1: tab probabilities
    # print('m: ', m)
    # print('t: ', t)
    t_full = np.reshape( t , [6,25] )
    all_binary_tabs = make_all_binary_tabs_for_binary_midi( m )
    print('len( all_binary_tabs): ', len(all_binary_tabs))
    if len(all_binary_tabs) == 0:
        print(m)
    all_probs = np.zeros( len(all_binary_tabs) )
    for i,b in enumerate( all_binary_tabs ):
        all_probs[i] = np.sum( b*t_full )
    return all_binary_tabs[ np.argmax( all_probs ) ]
# end midi_and_flat_tab_decision


def show_binary_matrices(bs, pause_time=0.1):
    for b in bs:
        plt.imshow(b, cmap='gray_r')
        plt.pause(pause_time)