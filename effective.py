# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:19:47 2014

@author: feiwu
"""
import sklearn
import cPickle as pickle
import numpy as np
from scipy import spatial
def c_vector(aa,bb):
    a = list()
    b = list()
    d = list()
    for key in aa.keys():
        if bb[key] != 0.5:
            a.append(aa[key])
            b.append(bb[key])
            d.append(0.5)
    return np.array(a),np.array(b), np.array(d)

def confidence_interval(result):
    return np.mean(np.array(result)), 1.96*np.std(np.array(result))

def list2vector(sig_value):
    sig_valuep = dict()
    for key in sig_value.keys():
        x = sig_value[key]
        sig_valuep[key] = sig_value[key][len(x)-1]
    #power_law_out= prefix + 'Label_sig_value_powerlaw.pickle'+ i
    bootstrap_out= prefix + 'Label_sig_value_bootstrap.pickle' + i
    return sig_valuep
gt_file = "gt.pickle"
prefix = '/home/feiwu/Desktop/540project/boot_uniform_rej/'
#prefix = '/home/feiwu/Desktop/540project/long_run/'

filename     = 'interp_result.pickle'
uniform_out  = 'sig_value_uniform.pickle'
power_law_out= 'sig_value_powerlaw.pickle'
bootstrap_out= 'sig_value_bootstrap.pickle'

uniform_result = list()
bootstrap_result = list()
power_law_result = list()
uniform_t = list()
bootstrap_t = list()
power_law_t = list()
power_samples= list()
for k in range(20):
    i = str(k)    
    uniform_out  = prefix + 'Label_sig_value_uniform.pickle' + i
    power_law_out= prefix +'Label_sig_value_powerlaw.pickle' + i 
    bootstrap_out= prefix + 'Label_sig_value_bootstrap.pickle' + i
    gt = pickle.load(open(gt_file))
    sig_value, samples,time_u = pickle.load(open(uniform_out))
    sig_value2,samples_p,time_p = pickle.load(open(power_law_out))
    sig_valueb,samples,time_b = pickle.load(open(bootstrap_out))
    ss = 0
    for key in samples_p:
        ss+= samples_p[key]
    power_samples.append(ss)
    #print time_u, time_p, time_b
    #rint one_boot
    value_pl = list2vector(sig_value2)
    value_bt = list2vector(sig_valueb)
    #sig_value2,
    #vec_sig, vec_gt,_= c_vector(sig_value,gt)
    vec_pl, vec_gt,_ = c_vector(value_pl, gt)
    vec_bt, vec_gt,dum = c_vector(value_bt, gt)
    vec_sig,vec_gt,_   = c_vector(sig_value, gt)
    #print vec_bt
    #print vec_pl
    #print vec_sig    
    #print vec_gt
   # result0 = 1 - spatial.distance.cosine(dum, vec_gt)
   # print "dum   sim:" + str(result0)
    
    result1 = 1 - spatial.distance.cosine(vec_sig, vec_gt)
    #print "uniform   sim:" + str(result1)
    uniform_result.append(result1)
    uniform_t.append(time_u)
    result2 = 1 - spatial.distance.cosine(vec_pl, vec_gt)
    power_law_result.append(result2)
    power_law_t.append(time_p)
    #print "power law sim:" + str(result2)
    result3 = 1 - spatial.distance.cosine(vec_bt, vec_gt)
    bootstrap_result.append(result3)
    bootstrap_t.append(time_b)
    #print "bootstrap sim:" + str(result3)
uniform_mean,unifom_std = confidence_interval(uniform_result)
boot_mean,boot_std      = confidence_interval(bootstrap_result)
power_mean,power_std  = confidence_interval(power_law_result)
uni_t_mean, uni_t_std =confidence_interval(uniform_t)  
boot_t_mean,boot_t_std    = confidence_interval(bootstrap_t)
power_t_mean,power_t_std  = confidence_interval(power_law_t)
power_samples_mean, power_samples_std = confidence_interval(power_samples)
print "Uniform :"
print uniform_mean,unifom_std
print "Boot:"
print boot_mean, boot_std
print "Power:"
print power_mean, power_std

print "Uniform T:"
print uni_t_mean/12,uni_t_std/12
print "Boot T:"
print boot_t_mean/12, boot_t_std/12
print "Power T:"
print power_t_mean/12, power_t_std/12
print "Power T samples:"
print power_samples_mean, power_samples_std