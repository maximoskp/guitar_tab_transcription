# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 07:35:13 2021

@author: user
"""

def get_filters_of_layer(l):
    w = l.weights[0]
    w_list = []
    for i in range(w.shape[3]):
        w_list.append( w[:,:,0,i].numpy() )
    return w_list # 1 is biases
# end get_filters_of_layer

