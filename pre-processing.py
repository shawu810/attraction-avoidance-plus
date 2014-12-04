# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 15:59:51 2014

@author: feiwu
Pre-processing
linear interpolation
"""
from utility import Trajectory

def load_raw_trajectories(filename):    
    current_id   = -1
    Trajectories = dict()  
    for line in open(filename,'r'):
        line_set = line.rstrip('\n').split(',')
        timestamp= int(line_set[0])
        lon      = float(line_set[1])
        lat      = float(line_set[2])
        object_id= int(line_set[3])
        if object_id != current_id:
            current_id = object_id    
            Trajectories[current_id] = Trajectory(object_id,list(),list())
        Trajectories[current_id].add_point(lat,lon,timestamp)
    return Trajectories

def compute_point(p1,p2,t1,t2,tc):
    lat1 = p1[0]
    lat2 = p2[0]
    lon1 = p1[1]
    lon2 = p2[1]
    ratio = float(tc-t1)/(t2-t1)
    lat_i = lat1 + (lat2-lat1)*ratio
    lon_i = lon1 + (lon2-lon1)*ratio
    return (lat_i, lon_i)

def find_later(timestamp, t_list, start = 0):
    for i in range(start-10,len(t_list)):
        if i < 0:
            continue
        if t_list[i] >= timestamp:
            return i
    return -1

def find_earlier(timestamp, t_list, start = 0):
    for i in reversed(range(0,start+10)):
        if i > len(t_list):
            continue
        if t_list[i] < timestamp:
            return i
    return 0

def interp(traj, min_timestamp, max_timestamp, sample_size ):
    current_timestamp = min_timestamp
    object_id         = traj.object_id
    current_samples   = 0
    traj_pointer     = find_later(current_timestamp,traj.timestamps)
    i_t_list = list()
    i_traj   = list()
    while ((current_timestamp < max_timestamp) and (current_samples < sample_size)):
        current_samples   += 1
        current_timestamp += gap_length
        i_t_list.append(current_timestamp)
        e_p = find_earlier(current_timestamp, traj.timestamps, traj_pointer)
        l_p = find_later(current_timestamp, traj.timestamps, traj_pointer)
        traj_pointer = l_p
        new_point = compute_point(traj.raw_points[e_p],traj.raw_points[l_p],traj.timestamps[e_p],traj.timestamps[l_p], current_timestamp)
        i_traj.append(new_point)
    return Trajectory(object_id,i_traj,i_t_list)
        
def interp_by_pair(trajectories):
    interp_trajs = dict()
    ranges       = len(trajectories)
    for one_t in range(ranges):
        for another_t in range(one_t+1,ranges):
            traj1 = trajectories[one_t]
            traj2 = trajectories[another_t]
            if traj1.object_id == traj2.object_id:
                continue
            keys = str(one_t) + "::" + str(another_t)
            print keys
            min_timestamp = max([traj1.timestamps[0],traj2.timestamps[0]])
            max_timestamp = min([traj1.timestamps[traj1.length-1],traj2.timestamps[traj2.length-2]])
            if min_timestamp >= max_timestamp:
                continue
            new_traj1     = interp(traj1, min_timestamp, max_timestamp, sample_size)
            new_traj2     = interp(traj2, min_timestamp, max_timestamp, sample_size)
            print "inter lenght:" + str(new_traj1.length) + '--' + str(new_traj2.length)
            interp_trajs[keys] = (new_traj1, new_traj2)
    return interp_trajs

global sample_size, gap_length    
filename     = 'dataset/data.csv'
output_file  = 'interp_result'
import cPickle as pickle
trajectories = load_raw_trajectories(filename)
gap_length   = 60*60 # one hour
sample_size  = 1000 
interp_result = interp_by_pair(trajectories)
pickle.dump(interp_result,open(output_file+'.pickle','w'))
    
        
    