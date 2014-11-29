# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:19:20 2014

@author: fei
"""
import numpy as np
import utility as ut

def point_shuffle(points):
    pass

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
        if ut.haversine(points1[i],points2[i]) < thres:
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

def compute_steps(points):
    distance_vector = list()
    for i in range(len(points)-1):
        pa = points[i]
        pb = points[i+1]
        dis = ut.haversine(pa,pb)
        distance_vector.append(dis)    
    return distance_vector

########################################### normal ##########################
import numpy as np
import scipy.stats
def fitting_random_walk_normal(points):
    dis_v = np.array(compute_steps(points))
    return np.mean(dis_v),np.std(dis_v),dis_v

def evaluate_traj_normal(dis_v, mu, std):
    mean, var, skew, kurt = scipy.stats.norm.stats(dis_v, moments='mvsk')
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 1)
    #ax.hist(dis_v)
    #fig.show()
    density = scipy.stats.norm(loc=mu,scale=std).pdf(dis_v)
    prob    = density/np.sum(density)
    return np.sum(np.log(prob))

##### 
import random
def compute_sig_value_Ga_random_walk(points1, points2, N):
    M = 5
    actual_meeting_freq = compute_meeting_freq(points1, points2)
    u_points= points2          
    mu2, std2, dis_v           = fitting_random_walk_normal(points2)
    prob_traj2          = evaluate_traj_normal(dis_v, mu2 , std2)             
    sig_count = 0
    count = 0
    total = 0
    while(1):
        shuffle(u_points)
        total += 1
        prob_shuffle = evaluate_traj_normal(u_points, mu2 , std2)
        log_ratio = (prob_shuffle - (M+prob_traj2))
        ratio    = prob_shuffle/(M*prob_traj2)
        #print ratio
        #print "ratio:"+str(log_ratio) + " prob_shuffle:"+str(prob_shuffle) + " prob_original:" + str(prob_traj2)
        if log_ratio < np.log(random.random()):
            continue  
        #print "accept"
        count += 1         
        meeting = compute_meeting_freq(points1, u_points)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
        if count >= N:
            break
    print "Gaussian:" +  str(total)
    return float(sig_count)/N
##############
    
def fitting_power_law(points):
    dis_v = np.array(compute_steps(points))
    N     = len(dis_v)
    return np.sum(np.log(dis_v))/N, dis_v
def evaluate_traj_zeta(dis_v, zeta):
    return np.sum(np.log(dis_v**-zeta))

from matplotlib.pylab import plot
from matplotlib.pyplot import show
def fitting_power_law2(points):
    dis_v = compute_steps(points)
    import powerlaw
    fit = powerlaw.Fit(dis_v, discrete=True)
    fig = powerlaw.plot_pdf(dis_v, color = 'b')
    return fit.power_law.alpha, fit.power_law.sigma, fit.xmin
    
def compute_sig_value_zeta_random_walk(points1, points2, N):
    M = 5
    actual_meeting_freq = compute_meeting_freq(points1, points2)
    u_points= points2          
    zeta2, dis_v          = fitting_power_law(points2)
    prob_traj2            = evaluate_traj_zeta(dis_v,zeta2)             
    sig_count = 0
    count = 0
    total = 0
    while(1):
        shuffle(u_points)
        total += 1
        prob_shuffle = evaluate_traj_zeta(compute_steps(u_points),zeta2)
        log_ratio = (prob_shuffle - (M+prob_traj2))
        ratio    = prob_shuffle/(M*prob_traj2)
        #print ratio
        #print "ratio:"+str(log_ratio) + " prob_shuffle:"+str(prob_shuffle) + " prob_original:" + str(prob_traj2)
        if ratio < random.random():
            continue  
        #print "accept"
        count += 1         
        meeting = compute_meeting_freq(points1, u_points)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
        if count >= N:
            break
    print "zeta:" + str(total)
    return float(sig_count)/N    

import numpy as np
import cPickle as pickle
global N, thres
from random import shuffle
N     = 500
thres = 100
filename = 'interp_result.pickle'
out_file = 'sig_value_uniform.pickle'
interp_result = pickle.load(open(filename,'r'))
#sig_result = compute_sig_value_all_pairs(interp_result)
#pickle.dump(sig_result, open(out_file, 'w'))
#sys.exit()
test_key = interp_result.keys()[1]
pair     = interp_result[test_key]
traj1    = pair[0]
traj2    = pair[1]

alpha,sigma,xmin = fitting_power_law2(traj2.raw_points)
zeta,_   =  fitting_power_law(traj2.raw_points)

import sys
sys.exit()
vu = compute_sig_value(traj1.raw_points, traj2.raw_points,N)
vg = compute_sig_value_Ga_random_walk(traj1.raw_points, traj2.raw_points,N)
vv = compute_sig_value_zeta_random_walk(traj1.raw_points, traj2.raw_points,N)

pickle.dump([vu,vg,vv], open('three values', 'w'))
############ compute the original sig value ##################


###7:38 running time


