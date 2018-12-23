import os
import sys
import glob
import numpy as np
import healpy as hp
import ccgpack as ccg
from scipy import stats

def p_value(lst):
    p_valu = []
    num = len(lst)
    for i in range(num):
        class1 = lst[i]
	pvs = []
	for j in range(num):
	    if i!=j:
	        class2 = lst[j]
                t , p = stats.ttest_ind(class1 , class2)
	        pvs.append(p)
        p_valu.append(np.max(pvs))
    return np.array(p_valu)

ccg.ch_mkdir('hh')  

gmulist = [0]+list(5*10**np.linspace(-8 , -5 , 10))
# IN CASE YOU WANT TO RUN IT parallel
#igmu = int(sys.argv[1])
#gmulist = gmulist[igmu:igmu+1]
ngmu = len(gmulist)


tadd = '../data/test_set/healpix_p/'
sim_name = tadd.split('/')[3]
res_add = '../data/classic/'+sim_name+'/'
ccg.ch_mkdir(res_add)

print('Buiding classical method results:')
#for j,gmu in enumerate(gmulist):
#    dir_name = '{:3.2e}'.format(gmu)
#    add = tadd+dir_name+'/'
#    stds = []
#    for i in range(480):
#        ccg.pop_percent(j*480+i,480*ngmu)
#        m = np.load(add+str(i)+'.npy')
#        m = ccg.curvelet(m,r_scale = 7)
#        m = ccg.filters(m, edd_method = 'sch')
#        std = np.std(m)
#        stds.append(std)
#    np.save(res_add+dir_name,stds)

lst1 = []
for gmu in gmulist:
    dir_name = '{:3.2e}'.format(gmu)
    var_file = res_add+dir_name+'.npy'
    lst1.append(np.load(var_file))
lst1 = np.array(lst1)    

result = p_value(lst1)
np.save(res_add[:-1],np.array([gmulist,result]))


tadd = '../data/test_set/ffp10_p/'
sim_name = tadd.split('/')[3]
res_add = '../data/classic/'+sim_name+'/'
ccg.ch_mkdir(res_add)

print('Buiding classical method results:')
for j,gmu in enumerate(gmulist):
    dir_name = '{:3.2e}'.format(gmu)
    add = tadd+dir_name+'/'
    stds = []
    for i in range(480):
        ccg.pop_percent(j*480+i,480*ngmu)
        m = np.load(add+str(i)+'.npy')
        m = ccg.curvelet(m,r_scale = 7)
        m = ccg.filters(m, edd_method = 'sch')
        std = np.std(m)
        stds.append(std)
    np.save(res_add+dir_name,stds)

lst1 = []
for gmu in gmulist:
    dir_name = '{:3.2e}'.format(gmu)
    var_file = res_add+dir_name+'.npy'
    lst1.append(np.load(var_file))
lst1 = np.array(lst1)    

result = p_value(lst1)
np.save(res_add[:-1],np.array([gmulist,result]))


