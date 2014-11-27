# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:19:20 2014

@author: fei
"""
import math
import operator,sys
import numpy as np
import time
from math import radians, cos, sin, asin, sqrt
########### untility ##################

def haversine(point1, point2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1 = point1[0]
    lon1 = point1[1]
    lat2 = point2[0]
    lon2 = point2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
   # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c 
    return km 

def compute_sig_value(points1, points2, N):
    actual_meeting_freq = compute_meeting_freq(points1, points2)
    u_points= points2                       
    N = 1000
    sig_count = 0
    for i in range(N):
        shuffle(u_points)
        meeting = compute_meeting_freq(points1, u_points)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
    return float(sig_count)/N
    
def compute_meeting_freq(points1,points2):
    total_number = len(points1)
    freq = 0
    for i in range(total_number):
        if haversine(points1[i],points2[i]) < thres:
            freq += 1
    return freq

def compute_sig_value_all_pairs(trajs):
    sig_result = dict()
    for f_key in trajs.keys():
        print f_key
        one_pair = interp_result[f_key]
        traj1 = one_pair[0]
        traj2 = one_pair[1]
        points1 = traj1.raw_points
        points2 = traj2.raw_points
        sig_value =  compute_sig_value(points1, points2, N)
        sig_result[f_key] = sig_value
    return sig_result

import numpy as np
import cPickle as pickle
global N, thres
from random import shuffle
N     = 1000
thres = 0.1
filename = 'interp_result.pickle'
out_file = 'sig_value_uniform.pickle'
interp_result = pickle.load(open(filename,'r'))
sig_result = compute_sig_value_all_pairs(interp_result)
pickle.dump(sig_result, open(out_file, 'w'))

###7:38 running time


