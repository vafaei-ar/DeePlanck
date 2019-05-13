import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import healpy as hp
import pylab as plt
from glob import glob
from ccgpack import sky2patch,ch_mkdir,pop_percent

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)
nside = 2048

def kelvin_check(m):
    if m.std()>1e-2:
        return 1e-6*m
    return m
    
def build_map(inmap,output,i,ntot,
                ns=1,g=0,sml=5.,
                lmax= 3*2048,prefix=''):    
    ch_mkdir(dest)
    dont = 1
    npatch = 12*ns**2
    for j in range(npatch):
        if not os.path.exists(output+prefix+str(i*npatch+j)+'.npy'):
            dont = 0
    if dont:
        return
    
    if sml!=0:    
        s = hp.read_map(inmap,nest=0,verbose=0)
        if g:
            s = kelvin_check(s)
        s = hp.smoothing(s,np.radians(sml/60.),lmax=lmax,verbose=0)
        s = hp.reorder(s , r2n=1)
    else:
        s = hp.read_map(inmap,nest=1,verbose=0)
    patches = sky2patch(s,ns)
    for j in range(npatch):
        pop_percent(i*npatch+j,ntot*npatch)
        np.save(output+prefix+str(i*npatch+j),patches[j])
        plt.imshow(patches[j], cmap=cmap)
        plt.savefig(output+prefix+str(i*npatch+j)+'.jpg')
        plt.close()
    

print('String maps:')
dest = '../data/string_p/'
ntot = 3
for i in range(ntot):
    inmap = '../data/string/map1n_allz_rtaapixlw_2048_'+str(i+1)+'.fits'
    build_map(inmap,dest,i,ntot)

#print('Healpix Gaussian maps:')
#dest = '../data/training_set/healpix_p/'
#ntot = 10
#for i in range(ntot):
#    inmap = '../data/healpix/map_2048_'+str(i)+'.fits'
#    build_map(inmap,dest,i,ntot,g=1)

print('FFP10 gaussian maps:')
dest = '../data/ffp10_p/'
ntot = 10
for i in range(ntot):
    inmap = '../data/ffp10/ffp10_lensed_scl_cmb_100_mc_'+str(i).zfill(4)+'.fits'
    build_map(inmap,dest,i,ntot,g=1)

print('OBSERVATIONS:') 
dest = '../data/observations_p/'
mlist = glob('../data/observations/*.fits')
for inmap in mlist:
    prefix = inmap.split('/')[-1][:-5]
    print('Observation: '+prefix)
    build_map(inmap,dest,i=0,ntot=1,g=1,prefix=prefix)

print('MASK:') 
dest = '../data/mask/'
inmap = '../data/mask/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
build_map(inmap,dest,i=0,ntot=1,sml=0)
