import matplotlib as mpl
mpl.use('agg')

import os
import sys
import glob
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

gmulist = [0]+list(5*10**np.linspace(-8 , -5 , 10))
ngmu = len(gmulist)
s2n = 10.

i_map = int(sys.argv[1])
i_srg = int(sys.argv[2])
i_gmup = int(sys.argv[3])

def blocker(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

if not os.path.exists('processed_images/gstd.npy'):
    gstd = []
    for i in range(42):
        patchs = []
        for j in range(12):
            patchs.append(np.load('../data/ffp10_p/'+str(i*12+j)+'.npy'))
        gstd.append(np.array(patchs).reshape(-1))
    gstd = np.mean(gstd)
    np.save('processed_images/gstd',gstd)
else:
    gstd = np.load('processed_images/gstd.npy')
    

#for i_patch in range(10):
#    i_mask = i_patch
#    maskp = np.load('../data/mask/'+str(i_mask)+'.npy')
#    mapp = np.load('../data/ffp10_p/'+str(i_map*12+i_patch)+'.npy')
#    srgp = np.load('../data/string_p/'+str(i_srg*12+i_stch)+'.npy')
#    
#    nsp = np.random.normal(0,gstd/s2n,mapp.shape)
#    ptch = mapp+gmulist[0]*srgp+nsp

#    mapp = np.ma.array(mapp, mask=maskp)
#    
#    mapp = blocker(mapp, 1024, 1024)

#    for i in range(4):
#        plt.imshow(mapp[i], cmap=cmap)
#        plt.savefig(str(4*i_patch+i)+'.jpg')
#        plt.close()

#ivp = 0
#for i_patch in range(12):
#    i_mask = i_patch
#    maskp = np.load('../data/mask/'+str(i_mask)+'.npy')    
#    maskp = blocker(maskp, 512, 512)
#    for i in maskp:
#        if 100.*np.sum([i==1])/np.prod(i.shape)>95:
#            ivp += 1
#print(ivp/48.)

add = 'processed_images/curveleted/'
ccg.ch_mkdir(add)  
res = {}

ctime = []
ftime = []
btime = []

t00 = time()

for i_gm in range(ngmu):
    if i_gm!=i_gmup:
        continue
    fname = add+'im{}_is{}_igmu{}'.format(i_map,i_srg,i_gm)
    
    if os.path.exists(fname+'.npy'):
        exit()
        
    for i_patch in range(12):
        
        gmu = gmulist[i_gm]
        i_stch = i_patch
        mapp = np.load('../data/ffp10_p/'+str(i_map*12+i_patch)+'.npy')
        srgp = np.load('../data/string_p/'+str(i_srg*12+i_stch)+'.npy')

        #nsp = np.random.normal(0,gstd/s2n,mapp.shape)
        ptch = mapp+gmu*srgp#+nsp
        t0 = time()
        ptch = blocker(ptch, 512, 512)
        btime.append(time()-t0)
        npp = len(ptch)

        for i in range(npp):
            mp = ptch[i]
            for rs in range(4,8):
                keyn = 'ip{}_ipp{}_c{}_'.format(i_patch,i,rs)
                t0 = time()
                mc = ccg.curvelet(mp,r_scale = rs)
                ctime.append(time()-t0)
                
                t0 = time()
                m1 = ccg.filters(mc, edd_method = 'sob')
                m2 = ccg.filters(mc, edd_method = 'sch')
                ftime.append(time()-t0)
                res[keyn+'sob'] = np.std(m1)
                res[keyn+'sch'] = np.std(m2)
        
    ccg.save(fname,res)

dtt = time()-t00    
print(np.mean(btime),np.mean(ctime),np.mean(ftime))
print(100.*np.sum(btime)/dtt,100.*np.sum(ctime)/dtt,100.*np.sum(ftime)/dtt)
exit()
# IN CASE YOU WANT TO RUN IT parallel
#igmu = int(sys.argv[1])
#gmulist = gmulist[igmu:igmu+1]
ngmu = len(gmulist)


ccg.ch_mkdir(res_add)

            

s = hp.read_map('../data/string/map1n_allz_rtaapixlw_2048_3.fits',nest=0,verbose=0)
s = hp.smoothing(s,np.radians(5./60),lmax= 3*2048,verbose=0)
s = hp.reorder(s , r2n=1)
sp = sky2patch(s,2)
nshape = sp[0].shape

print('Building test set:')

print('FFP10 maps:')
tadd = '../data/test_set/ffp10_p/'
add = '../data/ffp10/ffp10_lensed_scl_cmb_100_mc_'
ch_mkdir(tadd)  
for i in range(n_test):
    nsp = np.random.normal(0,gstd/s2n,nshape)
    ptch = gp[j]+gm*sp[j]+nsp
    np.save(tadd+dir_name+'/'+str(i*48+j),ptch)
    plt.imshow(ptch, cmap=cmap)
    plt.savefig(tadd+dir_name+'/'+str(i*48+j)+'.jpg')
    plt.close()
print('d!')
  














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






exit()
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


