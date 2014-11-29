# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:04:34 2014

@author: feiwu
"""

def get_all_distance(interp_result):
    all_dis = list()
    for f_key in interp_result.keys():
        pair  = interp_result[f_key]
        dis = ut.compute_steps(pair[0].raw_points) + ut.compute_steps(pair[1].raw_points)
        all_dis += dis
    return all_dis
import cPickle as pickle
import numpy as np
import utility as ut

filename = 'interp_result.pickle'
out_file = 'sig_value_uniform.pickle'
interp_result = pickle.load(open(filename,'r'))
#sig_result = compute_sig_value_all_pairs(interp_result)
#pickle.dump(sig_result, open(out_file, 'w'))
#sys.exit()
test_key = interp_result.keys()[8]
pair     = interp_result[test_key]
traj1    = pair[0]
traj2    = pair[1]

data1  = np.array(ut.compute_steps(traj1.raw_points))
from pylab import *
import powerlaw
Y = data1
X = np.array(get_all_distance(interp_result))
results = powerlaw.Fit(X, xmin = 1, xmax = 1000, linear_bins = True)
print results.power_law.alpha
print results.power_law.xmin
from powerlaw import plot_pdf, Fit, pdf
aa = results.power_law.plot_pdf(X,linestyle='--',color='b')
results.lognormal.plot_pdf(ax=aa,color='g', linestyle='--')
x,y = pdf(X, linear_bins = True)
x   = x[:-1]
aa.scatter(x,y, color='r',s=.5)
R, p = results.distribution_compare('lognormal', 'truncated_power_law',  normalized_ratio=True)
show()

edges, hist = powerlaw.pdf(Y)