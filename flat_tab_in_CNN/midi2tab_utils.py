# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 21:16:17 2021

@author: user
"""
import numpy as np
import itertools

def get_midi_full_fretboard():
    proper_tunning = np.array( [64, 59, 55, 50, 45, 40] )
    midi_full_fretboard = np.zeros( (6,25) )
    for i in range(25):
        midi_full_fretboard[:,i] = proper_tunning + i
    return midi_full_fretboard
# end get_midi_full_fretboard

def make_all_binary_tabs_for_binary_midi(m):
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
    # print('strings_for_notes: ', strings_for_notes)
    all_combinatios = list( itertools.product( *strings_for_notes ) )
    print('all_combinatios: ', all_combinatios)
    # keep only combinations with different elements
    combinations2keep = []
    for c in all_combinatios:
        # check for dublicates
        if len( list( set( c ) ) ) not in combinations2keep:
            combinations2keep.append( list(set( c )) )
    # construct binary fretboard for each string combination
    # print('combinations2keep: ', combinations2keep)
    all_binary_fretboards = []
    for c in combinations2keep:
        b = np.zeros( (6,25) )
        # get fret for each string
        for i,string in enumerate(c):
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
