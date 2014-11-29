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
    km = 6367 * c *1000
    return km 
    
def compute_steps(points):
    distance_vector = list()
    for i in range(len(points)-1):
        pa = points[i]
        pb = points[i+1]
        dis = haversine(pa,pb)
        distance_vector.append(dis)    
    return distance_vector
