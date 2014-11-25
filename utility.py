# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:44:28 2014

@author: feiwu

Preprocessing

"""


class Trajectory:
    def __init__(self, object_id, points=list()):
        self.object_id        = object_id
        self.raw_points       = points
        self.processed_points = dict() # key-value pair inteploated result
        self.index2time       = dict()
        
    def add_point(self, point):
        self.raw_points.append(point)
        
    def t2string(self):
        return "Trajectory: " + str(self.object_id) + " " + str(len(self.raw_points))
        
class Point:
    def __init__(self,lat,lon,timestamp):
        self.lat = lat
        self.lon = lon
        self.t   = timestamp
    def p2string(self):
        return "timestamp:" + str(self.t) + " lat:" + str(self.lat) + "  lon:" + str(self.lon)