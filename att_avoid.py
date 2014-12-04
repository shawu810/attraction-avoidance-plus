# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:19:20 2014

@author: fei
"""
import numpy as np
import utility as ut
import random

def point_shuffle(points, shuffle_size = 1):
    length        = len(points)
    new_points    = list(points)
    #print shuffle_size
    for i in range(shuffle_size):
        sampled_index = np.random.randint(length, size = 2)
        new_points[sampled_index[0]] = points[sampled_index[1]]
        new_points[sampled_index[1]] = points[sampled_index[0]]   
    return new_points
    
def point_close_shuffle(points, shuffle_size = 1):
    dis_v         = ut.compute_steps(points)
    length        = len(points)
    new_points    = list(points)
    #print shuffle_size
    for i in range(shuffle_size):
        sampled_index = np.random.randint(length, size = 2)
        new_points[sampled_index[0]] = points[sampled_index[1]]
        new_points[sampled_index[1]] = points[sampled_index[0]]   
    return new_points

def compute_sig_value(points1, points2, N):
    start_time = time()
    actual_meeting_freq = compute_meeting_freq(points1, points2)
    u_points   = list(points2)                       
    sig_count = 0
    for i in range(N):        
        shuffle(u_points)
        meeting = compute_meeting_freq(points1, u_points)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
    return float(sig_count)/N,N, time()-start_time

    
def compute_meeting_freq(points1,points2):
    total_number = len(points1)
    freq = 0
    for i in range(total_number):
        if ut.haversine(points1[i],points2[i]) < thres:
            freq += 1
    return freq

import mobility_model_fitting as model
from time import time
def compute_sig_value_all_pairs(trajs, method ='uniform', N = 1):
    #sig_result = dict()
    sig_lists  = dict()
    sig_length = dict()
    all_t =  0.0
    for f_key in trajs.keys():
        print f_key
        one_pair = interp_result[f_key]
        traj1 = one_pair[0]
        traj2 = one_pair[1]
        points1 = traj1.raw_points
        points2 = traj2.raw_points
        if method == 'uniform':
            sig_vlist,runs,t      =  compute_sig_value(points1, points2, N)
        elif method == 'powerlaw': 
            alpha,_               = model.power_law_par(interp_result)
            sig_vlist,runs,t      = compute_sig_value_reject_sampling(points1, points2, N)
        elif method == 'bootstrap':
            sig_vlist,runs,t = compute_sig_value_boot_strap(points1, points2, N)
        elif method == 'metropolis':
            sig_vlist,runs,t = compute_sig_value_metropolis(points1, points2, N*25)
        sig_length[f_key] = runs
        sig_lists[f_key]  = sig_vlist
        all_t += t
    return sig_lists,sig_length,all_t

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

def evaluate_traj_normal(dis_v,para):
    mu = para[0]
    std= para[1]
    mean, var, skew, kurt = scipy.stats.norm.stats(dis_v, moments='mvsk')
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 1)
    #ax.hist(dis_v)
    #fig.show()
    density = scipy.stats.norm(loc=mu,scale=std).pdf(dis_v)
    prob    = density/np.sum(density)
    return np.sum(np.log(prob))

 
##################################
    
def fitting_power_law(points):
    dis_v = np.array(compute_steps(points))
    N     = len(dis_v)
    return np.sum(np.log(dis_v))/N, dis_v
def evaluate_traj_zeta(dis_v, zeta):
    return np.sum(np.log(dis_v**-zeta))

def fitting_power_law2(points):
    dis_v = compute_steps(points)
    import powerlaw
    fit = powerlaw.Fit(dis_v, suppress_output = True,xmin = 1, xmax = 1000, linear_bins = True)
    return fit.power_law.alpha, fit.power_law.sigma, fit.xmin
 
def log_g_x_uniform(n,k):
    log_sum = 0
    for i in range(n-k+1,n):
        log_sum += np.log(i)
    return 1- log_sum
    
def log_f_x(fx,X, para):
    return fx(np.array(compute_steps(X)), para)

import levy 
def boot_strap(X, alpha):
    #import matplotlib.pyplot as plt
    #figure = plt.figure()
    x_cors  = [x[0] for x in X]
    y_cors  = [x[1] for x in X]
    x_min   = min(x_cors)-0.001
    x_max   = max(x_cors)+0.001
    y_min   = min(y_cors)-0.001
    y_max   = max(y_cors)+0.001
    xx      = ut.haversine([x_min,y_min],[x_max,y_min])
    yy      = ut.haversine([x_min,y_min],[x_min,y_max])
    #print "box size: " + str(xx) + " X " + str(yy)
    p_init  = np.mean(X, axis=0)
    x_init  = p_init[0]
    y_init  = p_init[1]
    boot_traj = list()   
    i = 0
    init_point = (x_init,y_init)
    while(i<len(X)):
        point   = levy.next_point(init_point,alpha)
        if(point[0]<x_min or point[0] > x_max or point[1]< y_min or point[1] > y_max):
            continue
        i += 1
        boot_traj.append(point)   
        init_point = point
    #xl = [x[0] for x in boot_traj]
    #yl = [x[1] for x in boot_traj]
    #plt.scatter(xl,yl,color='r')
    #plt.scatter(x_cors,y_cors,color='b')
    #figure.show()
    return boot_traj
    
def compute_sig_value_boot_strap(points1, points2, N):
    alpha1,_,_ = fitting_power_law2(points1)
    alpha2,_,_ = fitting_power_law2(points2)
    start_time = time()
    actual_meeting_freq = compute_meeting_freq(points1, points2)
    sig_count = 0.0
    count = 0
    sig_count_list = list()    
    while(1):
        X1 = boot_strap(points1,alpha1)
        X2 = boot_strap(points2,alpha2)
        count += 1         
        meeting = compute_meeting_freq(X1, X2)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
        sig_count_list.append(float(sig_count)/count)
        if count >= N:
            break
    return sig_count_list,N, time()- start_time
 
def compute_sig_value_metropolis(points1, points2, N_p):
    prefix = '/home/feiwu/Desktop/540project/'
    actual_meeting_freq =   compute_meeting_freq(points1, points2)       
    zeta2,_,_       = fitting_power_law2(points2)
    start_time      = time()  
    sig_count       = 0.0
    count = 0
    total = 0
    dis_count = 0
    sig_count_list = list()
    X        = list(points2)
    X        = point_shuffle(X,100)
    log_f_X  = log_f_x(evaluate_traj_zeta, X, zeta2 )
    flag     = True
    while(1):
        X_propose = point_shuffle(X,1)
        log_f_X_propose = log_f_x(evaluate_traj_zeta, X_propose, zeta2)
        log_u    = np.log(random.random())
        #print "ratio:" + str(log_f_X_propose - log_f_X) + "  log propose: " + str(log_f_X_propose) + "  log this: " + str(log_f_X)
        if log_f_X_propose - log_f_X > log_u:
            X_accept = X_propose
            #print "accpet"
            dis_count +=1
            meeting = compute_meeting_freq(points1, X_accept)
            flag    = False            
        else:
            X_accept = X  
        if flag:
            continue       
        count += 1
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
        sig_count_list.append(float(sig_count)/count)
        X = X_accept
        log_f_X = log_f_x(evaluate_traj_zeta, X_accept, zeta2 )
        if count >= N_p:
            break
        print "count" + str(count)
    np.savetxt(open( prefix + str(time()-start_time)+'result','w'),np.array(sig_count_list))
    return sig_count_list, total, time()- start_time  
        

def compute_sig_value_reject_sampling(points1, points2, N_p):
    M = 10
    actual_meeting_freq = compute_meeting_freq(points1, points2)        
    zeta2,_,_       = fitting_power_law2(points2)
    start_time      = time()
    #if params[0] == 'powerlaw':
    #    zeta2 = params[1]
    #mu, std, dis_v      = fitting_random_walk_normal(points2)
    #mu =58.5
    #std = 96.8
    #print zeta2
    #prob_traj2            = evaluate_traj_zeta(dis_v,zeta2)          
    sig_count = 0
    log_g_X   = log_g_x_uniform(len(points2),len(points2))
    #if params[0] == 'powerlaw':
    #log_g_X2   = log_f_x(evaluate_traj_zeta, points2, zeta2)   
    #print " gx2: " +str(log_g_X2)      
    e_size   =  int( 0.5*len(points2))
    count = 0
    total = 0
    sig_count_list = list()
    flag = True
    while(1):
        X = point_shuffle(points2,e_size)
        total += 1
        log_f_X = log_f_x(evaluate_traj_zeta, X, zeta2)      
        #log_f_X =  log_f_x( evaluate_traj_normal, X, [mu, std])        
        #prob_shuffle = evaluate_traj_zeta(compute_steps(u_points),zeta2)
        log_ratio = log_f_X - np.log(M) - log_g_X
        log_u    = np.log(random.random())
        #print "e_size:" + str(e_size) +" ratio:" + str(log_ratio) + " fx:"+str(log_f_X) + " gx:" + str(log_g_X) + " gx2: "
        #print "log_u:" + str(log_u)
        if log_ratio < log_u: 
            if e_size > 1 and flag:
                e_size -=1 
            #print "reject: " + str(e_size)
            continue 
        if flag:
            flag  = False
            e_size = max([e_size-10,1])
            start_time      = time()
            #log_g_X = log_g_x_uniform(len(points2),e_size)
            total = 0
        count += 1         
        meeting = compute_meeting_freq(points1, X)
        if meeting < actual_meeting_freq:
            sig_count += 1
        elif meeting == actual_meeting_freq:
            sig_count += 0.5
        sig_count_list.append(float(sig_count)/count)
        if count >= N_p:
            break
        print total
    return sig_count_list, total, time()- start_time 

import numpy as np
import cPickle as pickle
global N, thres
from random import shuffle
N     = 1000
thres = 100
LABEL  = True
prefix = '/home/feiwu/Desktop/540project/short_run/'

filename     = 'interp_result.pickle'
uniform_out  = 'sig_value_uniform.pickle'
power_law_out= 'sig_value_powerlaw.pickle'
power_metropolis_out= 'sig_value_metropolis.pickle'
bootstrap_out= 'sig_value_bootstrap.pickle'
interp_result = pickle.load(open(filename,'r'))
"""
test_key = interp_result.keys()[8]
pair     = interp_result[test_key]
traj1    = pair[0]
traj2    = pair[1]

#vm = compute_sig_value_metropolis(traj1.raw_points, traj2.raw_points,N)
vv, v_list1,t = compute_sig_value_reject_sampling(traj1.raw_points, traj2.raw_points,N)
sys.exit()
"""
gt_file_ni = "gt_no_imputation.pickle"
if LABEL:
    interp_result_label = dict()
    count = 0
    gtn   = pickle.load(open(gt_file_ni, 'r'))
    for f_key in gtn.keys():
        count +=1
        interp_result_label[f_key] = interp_result[f_key]  
    print len(interp_result_label)
    for i in range(20):
        sig_result_uniform,_,t1                      = compute_sig_value_all_pairs(interp_result_label, 'uniform',N)
        pickle.dump([sig_result_uniform,_,t1], open(prefix+"Label_"+uniform_out + str(i), 'w'))
    for i in range(20):
        sig_result_boot_strap,_,t2                   = compute_sig_value_all_pairs(interp_result_label, 'bootstrap',N)
        pickle.dump([sig_result_boot_strap,_,t2], open(prefix+"Label_"+bootstrap_out + str(i), 'w'))
    for i in range(20):
        sig_result_powerlaw,sig_length_powerlaw,t3   = compute_sig_value_all_pairs(interp_result_label, 'powerlaw',N)
        pickle.dump([sig_result_powerlaw,sig_length_powerlaw,t3], open(prefix+"Label_"+power_law_out+ str(i), 'w'))
else:
    for i in range(20):
        sig_result_uniform,_,t1                      = compute_sig_value_all_pairs(interp_result, 'uniform',N)
        pickle.dump([sig_result_uniform,_,t1], open(prefix+""+uniform_out + str(i), 'w'))
    for i in range(20):
        sig_result_boot_strap,_,t2                   = compute_sig_value_all_pairs(interp_result, 'bootstrap',N)
        pickle.dump([sig_result_boot_strap,_,t2], open(prefix+""+bootstrap_out + str(i), 'w'))
    for i in range(20):
        sig_result_powerlaw,sig_length_powerlaw,t3   = compute_sig_value_all_pairs(interp_result, 'powerlaw',N)
        pickle.dump([sig_result_powerlaw,sig_length_powerlaw,t3], open(prefix+""+power_law_out+ str(i), 'w'))
    #sig_result_powerlaw,sig_length_powerlaw,t4   = compute_sig_value_all_pairs(interp_result, 'powerlaw',N)




import sys
sys.exit()
""" this part of the code is for drawing individual figure"""
import matplotlib.pyplot as plt
vv, v_list1 = compute_sig_value_reject_sampling(traj1.raw_points, traj2.raw_points,N)
bt, v_list2 = compute_sig_value_boot_strap(traj1.raw_points, traj2.raw_points,N)
vu = compute_sig_value(traj1.raw_points, traj2.raw_points,N)
P = range(10)
shuffle_points = point_shuffle()
plt.plot(v_list1)
plt.show()

vg = compute_sig_value_Ga_random_walk()

pickle.dump([vu,vg,vv], open('three values', 'w'))
############ compute the original sig value ##################


###7:38 running time


