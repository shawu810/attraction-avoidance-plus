# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 16:09:16 2014

@author: http://code.google.com/p/randomwalks/source/browse/levy.py
"""

#!/usr/bin/env python

"""Draws a Levy flight using a Pareto distribution (started with PyGame stars.py example)"""

import random, math

#constants
#DMAX = 240
# Pareto shape parameter


def next_point(prev,ALPHA):
    angle = random.uniform(0,(2*math.pi))
#    angle = random.normalvariate(0,1.8)
    distance = 2 * random.paretovariate(ALPHA)/1100
#    distance = 2 * random.weibullvariate(1.0, 0.9)
    # cap distance at DMAX
#    if distance > DMAX:
#        distance = DMAX
    point = ((math.sin(angle) * distance)+prev[0], (math.cos(angle) * distance)+prev[1])
    return point

