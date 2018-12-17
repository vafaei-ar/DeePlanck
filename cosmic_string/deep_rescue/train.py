import matplotlib as mpl
mpl.use('agg')

import os
import sys
import numpy as np
import pylab as plt
import ngene as ng
from glob import glob
import ccgpack as ccg
import tensorflow as tf
from tqdm import tqdm, trange
from scipy.stats import ttest_ind
from random import choice,shuffle
from ngene.architectures.simple import architecture

nside = 2048
nx=100
ny=100
alpha = 4

alphas = []
success = []
dalpha = 0.05
pv_lim = 1e-10
training_epochs = 10
iterations=10
n_s = 50
learning_rate = 0.05

ntry = int(sys.argv[1])
n_layers = int(sys.argv[2])
filt = sys.argv[3]
#fig, (ax1,ax2)= plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
#ax1.imshow(x[0,:,:,0])
#ax1.axis('off')
#ax2.imshow(y[0,:,:,0])
#ax2.axis('off')

def standard(x):
    x = x-x.mean()
    x = x/x.std()
    return x

def get_slice(data,nx,ny):
    """Slice matrix in x and y direction"""
    lx,ly = data.shape  
    if nx==0 or nx==lx:
        slx = slice(0, lx)                
    else:
        idx = np.random.randint(0, lx - nx)            
        slx = slice(idx, (idx+nx))       
    if ny==0 or ny==ly:
        sly = slice(0, ly)                
    else:
        idy = np.random.randint(0, ly - ny)            
        sly = slice(idy, (idy+ny))
    return slx, sly

class DataProvider(object):
    def __init__(self,n_files,s_files,alpha,
                 nx=0,ny=0,n_buffer=10,reload_rate=10,filt=False):
        
#        self.l1 = l1
#        self.l2 = l2     
        self.n_files = n_files
        self.s_files = s_files
        
        nmin = min(len(n_files),len(s_files))
        if n_buffer>= nmin:
            n_buffer = nmin
            self.reload_rate = 0
            
        else:
            self.reload_rate = reload_rate
            
        self.nx,self.ny = nx,ny
        self.n_buffer = n_buffer
        self.alpha = alpha
        self.filt = filt
        self.counter = 0
        self.reload()
        
    def reload(self):
        print('Data provider is reloading...')
        self.n_set = []
        self.s_set = []
        self.d_set = []
        
        ninds = np.arange(len(self.n_files))
        sinds = np.arange(len(self.s_files))
        shuffle(ninds)
        shuffle(sinds)
        for i in range(self.n_buffer):
            filen = self.n_files[ninds[i]]
            files = self.s_files[sinds[i]]
            self.n_set.append(standard(np.load(filen)))
            signal = standard(np.load(files))
            self.s_set.append(signal)
            if self.filt:
                self.d_set.append(self.filt(signal))
            else:
                self.d_set.append(signal)
            

    def get_data(self): 
        self.counter += 1
        if self.reload_rate:
            if self.counter%self.reload_rate==0: self.reload() 
        n = choice(self.n_set)
        sind = choice(np.arange(self.n_buffer))
        s = self.s_set[sind]
        d = self.d_set[sind]
        return n,s,d
              

    def pre_process(self, n, s, d, alpha):
        nslice = get_slice(n,self.nx,self.ny)
        n = n[nslice]
        sslice = get_slice(s,self.nx,self.ny)
        s = s[sslice]
        sn = n + alpha*s
        d = d[sslice]
        sn,d = np.expand_dims(sn,-1),np.expand_dims(d,-1)
        return sn,d
    
    def __call__(self, n, alpha=None): 
    
        if alpha is None:
            alpha = self.alpha 
#        x,y = self.get_data()
        X = []
        Y = []
        for i in range(n):                
            n,s,d = self.get_data()
            sn,d = self.pre_process(n,s,d,alpha)
            X.append(sn)
            Y.append(d)
            
        X,Y = np.array(X),np.array(Y)
    
        return X,Y       

def arch(x_in):
    x_out = architecture(x_in=x_in,n_layers=n_layers,res=2)
    return x_out

def check(name,model,dp):
    l0 = []
    l1 = []
    for i in range(100):
        x,y = dp(1)
        x0,y = dp(1,0)
        l0.append(model.predict(x0).std())
        l1.append(model.predict(x).std())
    b0,h0 = ccg.pdf(l0,20)
    b1,h1 = ccg.pdf(l1,20)
    plt.plot(b0,h0)
    plt.plot(b1,h1)
    plt.savefig(name+'_pdf'+'.jpg')
    plt.close()
    
    fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,nrows=1,figsize=(15,7))
    x,y = dp(1)
    x_pred = model.predict(x)
    ax1.imshow(x[0,:,:,0])
    ax1.set_title(r'$G+\alpha S$')
    ax2.imshow(y[0,:,:,0])
    ax2.set_title(r'$S$')
    ax3.imshow(x_pred[0,:,:,0])
    ax3.set_title(r'$N+G+'+str(dp.alpha)+r'S$')
    plt.subplots_adjust(left=0.06, bottom=0.02, right=0.99, top=0.95)
    plt.savefig(name+'_sample'+'.jpg',dpi=100)
    plt.close()
    print('p-value:',ttest_ind(l0,l1)[1])
    return ttest_ind(l0,l1)[1]

x_files = glob('../data/training_set/healpix_p/*.npy')
y_files = glob('../data/training_set/string_p/*.npy')
filtf = filt=='ON'
if filtf:
    def filtf(x):
        return ccg.filters(x,edd_method='sch')
dp = DataProvider(x_files,y_files,alpha,nx=nx,ny=ny,
                  n_buffer=2,reload_rate=10000,filt=filtf)

model_add = './models/'+str(n_layers)+'_layers_f'+filt+'/'
res_dir = './results/'+str(n_layers)+'_layers_f'+filt+'/'
ccg.ch_mkdir(res_dir+'plots')
model = ng.Model(dp,restore=0,model_add=model_add+str(0),arch=arch)

print('# of variables:',model.n_variables)

if os.path.exists(res_dir+'info.npy'):
    i,dp.alpha,dalpha,learning_rate = np.load(res_dir+'info.npy')
    i = int(i)
    model.model_add=model_add+str(i)
    print('Loading model '+str(i)+' ...')
    model.restore()
else:
    i = 0

for _ in range(ntry):
    
    alphas.append(dp.alpha)
    model.model_add=model_add+str(i)
    print('Training model:{}, alpha:{}'.format(model.model_add,dp.alpha))
    model.train(data_provider=dp,training_epochs=training_epochs,
                        iterations=iterations,n_s=n_s,
                        learning_rate=learning_rate, time_limit=None,
                        metric=None, verbose=1,death_preliminary_check=30,
                        death_frequency_check=1000)
    learning_rate = learning_rate/1.02
    
    np.save(res_dir+'info',[i,dp.alpha,dalpha,learning_rate])
    pv = check(res_dir+'plots/'+str(i),model,dp)
    if pv>pv_lim and i!=0:
        dp.alpha = dp.alpha+dalpha
        if np.random.uniform()>0.5:
            dalpha = dalpha/1.5
        model.model_add=model_add+str(i-1)
        model.restore()
    else:
        while dp.alpha<=dalpha:
            dalpha = dalpha/1.5
        dp.alpha = dp.alpha-dalpha
        if np.random.uniform()>0.7:
            dalpha = dalpha*1.5
        i += 1  
    success.append(pv<pv_lim)        
