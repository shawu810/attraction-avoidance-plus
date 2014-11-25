# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:59:51 2014

@author: feiwu
"""
from utility import Trajectory,Point

def load_raw_trajectories(filename):    
    current_id   = 0
    Trajectories = dict() 
    one_trajectory = Trajectory(current_id,list())  
    for line in open(filename,'r'):
        line_set = line.rstrip('\n').split(',')
        timestamp= line_set[0]
        lon      = line_set[1]
        lat      = line_set[2]
        object_id= int(line_set[3])
        point    = Point(lat,lon,timestamp)
        if object_id != current_id:
            current_id = object_id    
            Trajectories[current_id] = one_trajectory
            one_trajectory = Trajectory(object_id,list())    
        one_trajectory.add_point(point)
    return Trajectories
        
        
filename     = 'dataset/data.csv'
trajectories = load_raw_trajectories(filename)