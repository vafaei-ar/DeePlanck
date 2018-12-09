import matplotlib as mpl
mpl.use('agg')

import os
import sys
import shutil
import argparse
import numpy as np
import pylab as plt
import healpy as hp
from ccgpack import sky2patch,ch_mkdir,download
#from healpy import cartview

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nsim', action="store", type=int, default=20)
args = parser.parse_args()
replace = args.r
n_gaussian = args.nsim

nside = 2048
lmax = 2*nside

if nside==2048:
    n_string=3
    ex = 'gz'
    import gzip
    
    def extract(in_file,out_file):
        with gzip.open(in_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
elif nside==4096:
    n_string=1
    ex = 'xz'
        
    def extract(in_file,out_file):
        os.system('unxz '+in_file)   
else:
    assert 0,'Nside has to be either 2048 or 4096!'

#n_gaussian = 10
#nside = 2048
#lmax = 3500
#fwhm = float(sys.argv[1])

cl = np.load('./data/cl_planck_lensed.npy')
ll = cl[:lmax,0]
cl = cl[:lmax,1]

ch_mkdir('./data/healpix/') 

for i in range(n_gaussian):

    if not os.path.exists('./data/healpix/'+'map_'+str(nside)+'_'+str(i)+'.fits') or replace:
        print('Simulation gaussian healpix map: '+str(i))

        m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=0, sigma=None, new=1, verbose=0)
        cl_map = hp.sphtfunc.alm2cl(alms)

        hp.mollview(m, nest=0, cmap=cmap)
        hp.write_map('./data/healpix/'+'map_'+str(nside)+'_'+str(i)+'.fits', m, overwrite=1)
        plt.savefig('./data/healpix/'+'map_'+str(nside)+'_'+str(i)+'.jpg')
        plt.close()
            
        plt.figure(figsize=(10,6))
        dl1 = []
        dl2 = []
        for j in range(ll.shape[0]):
                dl1.append(ll[j]*(ll[j]+1)*cl[j]/(2*np.pi))
                dl2.append(ll[j]*(ll[j]+1)*cl_map[j]/(2*np.pi))

        plt.plot(ll,dl2,'r--',label='Simulation')
        plt.plot(ll,dl1,'b--',lw=2,label='Orginal')
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(labelsize=15)
        plt.xlabel(r'$\ell$',fontsize=25)
        plt.ylabel(r'$D_{\ell}$',fontsize=25)

        plt.legend(loc='best',fontsize=20)
        plt.savefig('./data/healpix/power_'+str(nside)+'_'+str(i)+'.jpg')
        plt.close()
                
ch_mkdir('./data/string/') 

for i in range(n_string): 
    strnum = str(i+1)
    if nside==4096:
        strnum = strnum+'b'
    if not os.path.exists('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex):
        print('Downloading string: '+str(i))
        download('http://cp3.irmp.ucl.ac.be/~ringeval/upload/data/'+str(nside)+'/map1n_allz_rtaapixlw_'+str(nside)+'_'+strnum+'.fits.'+ex,
          './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex)

    load_status = 0
    if not os.path.exists('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'):
        print('Extracting string: '+str(i))
#        with gzip.open('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex, 'rb') as f_in:
#            with open('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits', 'wb') as f_out:
#                shutil.copyfileobj(f_in, f_out)
        in_file = './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex
        out_file = './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'
        extract(in_file,out_file)
        
        ss = hp.read_map('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits',verbose=0,nest=1)    
        load_status = 1        
        hp.mollview(ss, nest=1, cmap=cmap)    
        plt.savefig('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.jpg')
        plt.close()
 
ch_mkdir('./data/ffp10/') 
                 
for i in range(n_gaussian):
    mm = str(i).zfill(4)
    
    if not os.path.exists('./data/ffp10/ffp10_lensed_scl_cmb_100_mc_'+mm+'.fits'):
        download('http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_100_mc_'+mm+'.fits',
                './data/ffp10/ffp10_lensed_scl_cmb_100_mc_'+mm+'.fits')
                
        ss = hp.read_map('./data/ffp10/ffp10_lensed_scl_cmb_100_mc_'+mm+'.fits',verbose=0,nest=1)    
        hp.mollview(ss, nest=1, cmap=cmap)              
        plt.savefig('./data/ffp10/ffp10_lensed_scl_cmb_100_mc_'+mm+'.jpg')
        plt.close()        
                
            
            
            
     
