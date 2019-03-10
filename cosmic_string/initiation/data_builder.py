import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import healpy as hp
import pylab as plt
from ccgpack import sky2patch,ch_mkdir,pop_percent

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

def kelvin_check(m):
    if m.std()>1e-2:
        return 1e-6*m
    return m

nside = 2048
n_gaussian = 4
n_gaussian = n_gaussian//2

print('Building training set:')
print('String maps:')
ch_mkdir('../data/training_set/string_p')
for i in range(2):

    dont = 1
    for j in range(12):
        if not os.path.exists('../data/training_set/string_p/'+str(i*12+j)+'.npy'):
            dont = 0
    if dont:
        continue

    s = hp.read_map('../data/string/map1n_allz_rtaapixlw_2048_'+str(i+1)+'.fits',nest=0,verbose=0)
    s = hp.smoothing(s,np.radians(5./60),lmax= 3*2048,verbose=0)
    s = hp.reorder(s , r2n=1)
    patches = sky2patch(s,1)
    for j in range(12):
        pop_percent(i*12+j,2*12)
        np.save('../data/training_set/string_p/'+str(i*12+j),patches[j])
        plt.imshow(patches[j], cmap=cmap)
        plt.savefig('../data/training_set/string_p/'+str(i*12+j)+'.jpg')
        plt.close()
print('completed!')

print('Healpix Gaussian maps:')
ch_mkdir('../data/training_set/healpix_p')
add = '../data/healpix/map_2048_'
for i in range(n_gaussian):

    dont = 1
    for j in range(12):
        if not os.path.exists('../data/training_set/healpix_p/'+str(i*12+j)+'.npy'):
            dont = 0
    if dont:
        continue

    mm = str(i)
    g = hp.read_map(add+mm+'.fits',nest=0,verbose=0)
    g = kelvin_check(g)
    g = hp.smoothing(g,np.radians(5./60),lmax= 3*2048,verbose=0)
    g = hp.reorder(g , r2n=1)
    patches = sky2patch(g,1)
    for j in range(12):
        pop_percent(i*12+j,n_gaussian*12)
        np.save('../data/training_set/healpix_p/'+str(i*12+j),patches[j])
        plt.imshow(patches[j], cmap=cmap)
        plt.savefig('../data/training_set/healpix_p/'+str(i*12+j)+'.jpg')
        plt.close()
print('d!')

print('FFP10 gaussian maps:')
ch_mkdir('../data/training_set/ffp10_p')        
add = '../data/ffp10/ffp10_lensed_scl_cmb_100_mc_'
for i in range(n_gaussian):

    dont = 1
    for j in range(12):
        if not os.path.exists('../data/training_set/ffp10_p/'+str(i*12+j)+'.npy'):
            dont = 0
    if dont:
        continue

    mm = str(i).zfill(4)
    g = hp.read_map(add+mm+'.fits',nest=0,verbose=0)
    g = kelvin_check(g)
#    g = hp.smoothing(g,np.radians(5./60),lmax= 3*2048,verbose=0)
    g = hp.reorder(g , r2n=1)
    patches = sky2patch(g,1)
    for j in range(12):
        pop_percent(i*12+j,n_gaussian*12)
        np.save('../data/training_set/ffp10_p/'+str(i*12+j),patches[j])
        plt.imshow(patches[j], cmap=cmap)
        plt.savefig('../data/training_set/ffp10_p/'+str(i*12+j)+'.jpg')
        plt.close()
print('d!')

#########################################################
            
gmulist = list(5*10**np.linspace(-8 , -5 , 10))+[0]
ngmu = len(gmulist)
s2n = 10.

s = hp.read_map('../data/string/map1n_allz_rtaapixlw_2048_3.fits',nest=0,verbose=0)
s = hp.smoothing(s,np.radians(5./60),lmax= 3*2048,verbose=0)
s = hp.reorder(s , r2n=1)
sp = sky2patch(s,2)
nshape = sp[0].shape

print('Building test set:')
print('Healpix maps:')
tadd = '../data/test_set/healpix_p/'   
add = '../data/healpix/map_2048_'
ch_mkdir(tadd)  
for i in range(n_gaussian):

    dont = 1
    for k,gm in enumerate(gmulist):
        dir_name = '{:3.2e}'.format(gm)
        for j in range(12):
            if not os.path.exists(tadd+dir_name+'/'+str(i*48+j)+'.npy'):
                dont = 0
    if dont:
        continue

    mm = str(i+n_gaussian)
    g = hp.read_map(add+mm+'.fits',nest=0,verbose=0)
    g = kelvin_check(g)
    gstd = g.std()
    g = hp.smoothing(g,np.radians(5./60),lmax= 3*2048,verbose=0)
    g = hp.reorder(g , r2n=1)
    gp = sky2patch(g,2)
    for k,gm in enumerate(gmulist):
        dir_name = '{:3.2e}'.format(gm)
        ch_mkdir(tadd+dir_name)
        for j in range(48):
            pop_percent(i*ngmu*48+k*48+j,n_gaussian*ngmu*48)
            nsp = np.random.normal(0,gstd/s2n,nshape)
            ptch = gp[j]+gm*sp[j]+nsp
            np.save(tadd+dir_name+'/'+str(i*48+j),ptch)
            plt.imshow(ptch, cmap=cmap)
            plt.savefig(tadd+dir_name+'/'+str(i*48+j)+'.jpg')
            plt.close()
print('d!')

print('FFP10 maps:')
tadd = '../data/test_set/ffp10_p/'
add = '../data/ffp10/ffp10_lensed_scl_cmb_100_mc_'
ch_mkdir(tadd)  
for i in range(n_gaussian):

    dont = 1
    for k,gm in enumerate(gmulist):
        dir_name = '{:3.2e}'.format(gm)
        for j in range(12):
            if not os.path.exists(tadd+dir_name+'/'+str(i*48+j)+'.npy'):
                dont = 0
    if dont:
        continue
        
    mm = str(i+n_gaussian).zfill(4)
    g = hp.read_map(add+mm+'.fits',nest=0,verbose=0)
    g = kelvin_check(g)
    gstd = g.std()
    g = hp.smoothing(g,np.radians(5./60),lmax= 3*2048,verbose=0)
    g = hp.reorder(g , r2n=1)
    gp = sky2patch(g,2)
    for k,gm in enumerate(gmulist):
        dir_name = '{:3.2e}'.format(gm)
        
        dont = 1
        for j in range(12):
            if not os.path.exists(tadd+dir_name+'/'+str(i*48+j)+'.npy'):
                dont = 0
        if dont:
            continue
        
        ch_mkdir(tadd+dir_name)
        for j in range(48):
            pop_percent(i*ngmu*48+k*48+j,n_gaussian*ngmu*48)
            nsp = np.random.normal(0,gstd/s2n,nshape)
            ptch = gp[j]+gm*sp[j]+nsp
            np.save(tadd+dir_name+'/'+str(i*48+j),ptch)
            plt.imshow(ptch, cmap=cmap)
            plt.savefig(tadd+dir_name+'/'+str(i*48+j)+'.jpg')
            plt.close()
print('d!')
  
print('MASK:') 
dont = 1
for i in range(48):
    if not os.path.exists('../data/mask/'+str(i)+'.npy'):
        dont = 0
if dont==0:
    mask = hp.read_map('../data/mask/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits',nest=1,verbose=0)
    maskp = sky2patch(mask,2)
    for i in range(48):
        pop_percent(i,48)
        np.save('../data/mask/'+str(i),maskp[i])
        plt.imshow(maskp[i], cmap=cmap)
        plt.savefig('../data/mask/'+str(i)+'.jpg')
        plt.close()
    print('d!')
  
