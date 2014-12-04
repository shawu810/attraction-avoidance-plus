# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 14:40:37 2014

@author: feiwu
"""

import cPickle as pickle

uniform_file = "sig_value_uniform.pickle"

results,_,_   = pickle.load(open(uniform_file))
all_pairs = results.keys()


id_mapping = {0:87,1:58,2:57,3:83,4:85, 5:55,6:84,7:83,8:54,9:53,10:52,11:51}
id_rev_map = {87:0,58:1,57:2,83:3,85:4, 55:5,84:6,83:7,54:8,53:9,52:10,51:11}
groups     = [(51,87),(52,53),(54,55),(57,58),(83,84),(85,86)]
av_groups  = [(52,83),(53,84),(52,84),(53,83)]
gt         = dict()
gt_nimp    = dict()
for one_key in all_pairs:
    ids = one_key.split("::")
    key1 = id_mapping[int(ids[0])]
    key2 = id_mapping[int(ids[1])]
    if (key1,key2) in groups or (key2,key1) in groups:
        gt[one_key] = 1
        gt_nimp[one_key] = 1
    elif (key1,key2) in av_groups or (key2,key1) in av_groups:
        gt[one_key] = 0
        gt_nimp[one_key] = 0
    else:
        gt[one_key] = 0.5


gt_file = "gt.pickle"
gt_file_ni = "gt_no_imputation.pickle"

filename     = 'interp_result.pickle'
uniform_out  = 'sig_value_uniform.pickle'
power_law_out= 'sig_value_powerlaw.pickle'
bootstrap_out= 'sig_value_bootstrap.pickle'

pickle.dump(gt, open(gt_file,'w'))
pickle.dump(gt_nimp, open(gt_file_ni,'w'))