# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:44:28 2014

@author: feiwu

Preprocessing

"""
    

class Trajectory:
    def __init__(self, object_id, points=list(), timestamps=list()):
        self.object_id        = object_id
        self.raw_points       = points
        self.timestamps       = timestamps
        self.length           = len(points)
        
    def add_point(self, lat, lon, timestamp):
        self.raw_points.append((lat,lon))
        self.timestamps.append(timestamp)
        self.length += 1
        
    def t2string(self):
        return "Trajectory: " + str(self.object_id) + " " + str(len(self.raw_points))
        
#point lat, lon