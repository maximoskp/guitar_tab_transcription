# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:00:08 2021

@author: user
"""

import numpy as np

def get_64_fingering_weights():
    w = np.zeros( (6,6,1,64) )
    i = 0
    # single string
    for j in range(6):
        w[i,0,0, i] = 1
        i += 1
    # bass 6 minor ===============================================
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    w[4,2,0, i] = 1
    w[5,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3-chord
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    
    # bass 6 major =================================================
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,2,0, i] = 1
    w[4,2,0, i] = 1
    w[5,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    i += 1
    # 3-chord
    w[2,1,0, i] = 1
    w[3,2,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    
    # bass 5 minor ================================================
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    w[2,2,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    w[2,2,0, i] = 1
    i += 1
    # 3 mid
    w[1,1,0, i] = 1
    w[2,2,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    
    # bass 5 major =================================================
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,2,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,2,0, i] = 1
    i += 1
    # 3 mid
    w[1,2,0, i] = 1
    w[2,2,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    
    # bass 6 minor 7th ===============================================
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    w[3,0,0, i] = 1
    w[4,2,0, i] = 1
    w[5,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    # 4high-alt
    w[0,0,0, i] = 1
    w[1,3,0, i] = 1
    w[2,0,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,3,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3-chord
    w[2,0,0, i] = 1
    w[3,0,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    
    # bass 6 major dom7 =================================================
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    w[4,2,0, i] = 1
    w[5,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    # 4high-alt
    w[0,0,0, i] = 1
    w[1,3,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,3,0, i] = 1
    w[2,1,0, i] = 1
    i += 1
    # 3-chord
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    
    # bass 6 major maj7 =================================================
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,1,0, i] = 1
    w[4,2,0, i] = 1
    w[5,0,0, i] = 1
    i += 1
    # 4high
    w[0,0,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,1,0, i] = 1
    i += 1
    # 4high-quartal
    w[0,2,0, i] = 1
    w[1,2,0, i] = 1
    w[2,1,0, i] = 1
    w[3,1,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    w[2,1,0, i] = 1
    i += 1
    # 3-chord
    w[2,1,0, i] = 1
    w[3,1,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    
    # bass 5 minor 7th ================================================
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3high-alt
    w[0,3,0, i] = 1
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3 mid
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    
    # bass 5 major dom7 =================================================
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3high-alt
    w[0,3,0, i] = 1
    w[1,2,0, i] = 1
    w[2,2,0, i] = 1
    i += 1
    # 3 mid
    w[1,2,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    
    # bass 5 major maj7 =================================================
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,1,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    w[2,1,0, i] = 1
    i += 1
    # 3high-alt
    w[0,4,0, i] = 1
    w[1,2,0, i] = 1
    w[2,2,0, i] = 1
    i += 1
    # 3 mid
    w[1,2,0, i] = 1
    w[2,1,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    
    # bass 5 minor 6th ================================================
    w[0,2,0, i] = 1
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    w[4,0,0, i] = 1
    i += 1
    # 3high
    w[0,2,0, i] = 1
    w[1,1,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3high-alt
    w[0,1,0, i] = 1
    w[1,2,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    # 3high-alt
    w[0,2,0, i] = 1
    w[1,1,0, i] = 1
    w[2,2,0, i] = 1
    i += 1
    # 3 mid
    w[1,1,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    # 4 high
    w[0,1,0, i] = 1
    w[1,1,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    # 4 high-alt
    w[0,2,0, i] = 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    w[3,0,0, i] = 1
    i += 1
    
    # 2-chords maj
    w[0,0,0, i] = 1
    w[1,1,0, i] = 1
    i += 1
    w[1,0,0, i] = 1
    w[2,0,0, i] = 1
    i += 1
    w[2,0,0, i] = 1
    w[3,1,0, i] = 1
    i += 1
    w[3,0,0, i] = 1
    w[4,1,0, i] = 1
    i += 1
    w[4,0,0, i] = 1
    w[5,1,0, i] = 1
    i += 1
    
    # 2-chords min
    w[0,0,0, i] = 1
    w[1,2,0, i] = 1
    i += 1
    w[1,0,0, i] = 1
    w[2,1,0, i] = 1
    i += 1
    w[2,0,0, i] = 1
    w[3,2,0, i] = 1
    i += 1
    w[3,0,0, i] = 1
    w[4,2,0, i] = 1
    i += 1
    w[4,0,0, i] = 1
    w[5,2,0, i] = 1
    i += 1
    
    # random
    w[:,:,0,i] = np.random.rand(6,6)
    
    print('total fixed filters: ' + str(i))
    
    return w

def get_128_fingering_weights():
    w = np.zeros( (6,6,1,128) )
    i = 64
    
    # first 64 filters
    w[:,:,:,:i] = get_64_fingering_weights()
    
    # random
    w[:,:,:,i:] = np.random.rand(6,6,1,64)
    
    return w