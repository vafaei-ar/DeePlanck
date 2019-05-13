import matplotlib as mpl
mpl.use('agg')

import os
import sys
from glob import glob
from time import time
import numpy as np
import pylab as plt
import ccgpack as ccg
from scipy import stats

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',0.)

add = 'processed_images/'
ccg.ch_mkdir(add) 

add = 'processed_images/curveleted/' 
if os.path.exists(add+'observations.npy'):
    exit()
ccg.ch_mkdir(add)

def blocker(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

res = {}

masked_per = []
for i_patch in range(12):
    i_mask = i_patch
    maskp = np.load('../data/mask/'+str(i_mask)+'.npy')    
    maskp = blocker(maskp, 512, 512)
    pp = []
    for i in maskp:
        pp.append(100.*np.sum([i==1])/np.prod(i.shape))
    masked_per.append(pp)
        
res['masked_per'] = masked_per     

mlist = glob('../../data/observations/*.fits')
for inmap in mlist:
    prefix = inmap.split('/')[-1][:-5]
    for i_patch in range(12):
        dest = '../data/observations_p/'
        ptch = np.load(dest+prefix+str(i_patch)+'.npy')
        ptch = blocker(ptch, 512, 512)
        npp = len(ptch)
        for i in range(npp):
            mp = ptch[i]
            for rs in range(4,8):
                keyn = '{}_ip{}_ipp{}_c{}_'.format(prefix,i_patch,i,rs)
                mc = ccg.curvelet(mp,r_scale = rs)
                m1 = ccg.filters(mc, edd_method = 'sob')
                m2 = ccg.filters(mc, edd_method = 'sch')
                res[keyn+'sob'] = np.std(m1)
                res[keyn+'sch'] = np.std(m2)
        
ccg.save(add+'observations',res)








