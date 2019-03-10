import matplotlib as plt
plt.use('agg')

import sys
import numpy as np
import ngene as ng
import pylab as plt
import ccgpack as ccg
from glob import glob
import tensorflow as tf
from random import choice,shuffle
from matplotlib.colors import LogNorm

#print( ' *cnn* :  cnn without any dropout , with kernel size = 5, filters =36, iters = 300 , Gu:[1e-9 , 1e-6]. applied on ffp10 simulations ' )

n_conv = int(sys.argv[1])
dofilt = sys.argv[2]

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
    def __init__(self,n_files,s_files,gmus,
                 nx=0,ny=0,n_buffer=10,
                 reload_rate=100,filt=None):
        
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
        self.gmus = gmus
        if filt is None:
            def filt(x):
                return x
        self.filt = filt
        self.counter = 0
        self.reload()
        
    def reload(self):
        print('Data provider is reloading...')
        self.n_set = []
        self.s_set = []
#        self.d_set = []
        
        ninds = np.arange(len(self.n_files))
        sinds = np.arange(len(self.s_files))
        shuffle(ninds)
        shuffle(sinds)
        for i in range(self.n_buffer):
            filen = self.n_files[ninds[i]]
            files = self.s_files[sinds[i]]
            self.n_set.append(np.load(filen))
            signal = np.load(files)
            self.s_set.append(signal)
#            if self.filt:
#                self.d_set.append(self.filt(signal))
#            else:
#                self.d_set.append(signal)
#            

    def get_data(self): 
        self.counter += 1
        if self.reload_rate:
            if self.counter%self.reload_rate==0: 
                self.reload() 
        n = choice(self.n_set)
        sind = choice(np.arange(self.n_buffer))
        s = self.s_set[sind]
#        d = self.d_set[sind]
        return n,s#,d
              

    def pre_process(self, n, s, gmu):
        nslice = get_slice(n,self.nx,self.ny)
        n = n[nslice]
        sslice = get_slice(s,self.nx,self.ny)
        s = s[sslice]
        sn = n + gmu*s
        sn = self.filt(sn)
#        d = d[sslice]
        sn = np.expand_dims(sn,-1)
#        d = np.expand_dims(d,-1)
        return sn#,d
    
    def __call__(self, n, gmus=None): 
    
        if gmus is None:
            gmus = self.gmus
#        x,y = self.get_data()
        X = []
        Y = []
        for i in range(n):                
            n,s = self.get_data()
            gmu = choice(gmus)
            sn = self.pre_process(n,s,gmu)
            X.append(sn-sn+gmu)
            Y.append(-np.log(gmu+1e-30))
            
        X = np.array(X)
        Y = np.array(Y)
    
        return X,Y#[:,None]

def arch_maker(x,n_conv):
    #x_in = tf.placeholder(tf.float32,[None,nx,ny,n_channel])
    #y_true = tf.placeholder(tf.float32,[None , n_channel])
    #learning_rate = tf.placeholder(tf.float32)
    
    for _ in range(n_conv):
        x = tf.layers.conv2d(x,filters=16,kernel_size=5,
                              strides=(1, 1),padding='same',
                              activation=tf.nn.relu)
        x = tf.layers.average_pooling2d(x,pool_size=2,strides=2)
        print(x)

    x = tf.contrib.layers.flatten(x)
    print(x)
    x = tf.layers.dense(x, 10 , activation=tf.nn.relu)
    print(x)
    y = tf.layers.dense(x, 1 , activation=tf.nn.relu)
    print(x)
    return y
    
def arch(x):
    return arch_maker(x,n_conv)
    
#def loss(y_true,x_out):
#    return 1e3*tf.reduce_mean(tf.pow(y_true-x_out,2))

training_epochs = 10
iterations=10
n_s = 50
learning_rate = 0.05

g_files = sorted(glob('./data/training_set/healpix_p/*.npy'))
s_files = sorted(glob('./data/training_set/string_p/*.npy'))

if len(g_files)*len(s_files)==0:
    print('Somthing is wrong with initiation.')
    exit()

if dofilt[0]=='y':
    def filt(x):
        return ccg.filters(x,edd_method='sch')
else:
    filt = None

#omin,omax = np.log(self.gmb[0]),np.log(self.gmb[1])
#gmus = np.exp(np.random.uniform(omin,omax,n))
gmus = [0]+list(5*10**np.linspace(-8 , -5 , 10))
dp = DataProvider(g_files,s_files,
                  gmus=gmus,
                  nx=50,ny=50,n_buffer=10,
                  reload_rate=10e5,filt=filt)
                  
#x,y = dp(4,gmus=np.array(4*[1e-3]))
#print x.shape
#print y.shape
#fig,ax=plt.subplots(1,1,figsize=(5,5))
#ax.imshow(x[0,:,:,0],norm=LogNorm(),cmap=plt.get_cmap('jet'))
#plt.title('G + Gu*S')
#plt.savefig('x_lognorm ')
#fig,ax=plt.subplots(1,1,figsize=(5,5))
#ax.imshow(x[0,:,:,0])
#plt.title('G + Gu*S')
#plt.savefig('x')
#exit()

model_add='./model/'+str(n_conv)+'_layers_'+dofilt+'/'
model = ng.Model(dp,restore=1,
                 model_add=model_add,
                 arch=arch)#,loss=loss)

learning_rate = learning_rate/(1.02)**1000
for ii in range(4000):
    
    model.train(data_provider=dp,training_epochs=training_epochs,
                        iterations=iterations,n_s=n_s,
                        learning_rate=learning_rate, verbose=1)
    learning_rate = learning_rate/1.02

#x,y = dp_total(1000)

#pred = sess.run(y_out, feed_dict={x_in: x})
#d=abs(pred-y)/y
#delta=np.mean(d)

#print('accuracy =' , 100*(1-delta))


#r_squared = 1 - ( np.sum((y-pred)**2)/ np.sum((y-np.mean(y))**2) )
#print('r_squared =' ,r_squared)

#measurable= (1-r_squared) * (1e-6 - 1e-9)
#print('min_measurable=' , measurable)

#"""
#plt.loglog(y , y , 'r--' , label='Gu_pred = Gu_fid')
#plt.axvline(x = measurable*1e9 , color='g' , label= 'min measurable')
#plt.loglog(y , pred,'b.')
#plt.xlabel('Gu_fid')
#plt.ylabel('Gu_pred')
#plt.title('ff10')
##plt.legend(bbox_to_anchor=(1.05, 1),loc =2 , borderaxespad=0.)
#plt.savefig( '1' ,  bbox_inches='tight')

#"""

#end_time = time.time()
#print('duration=' , timedelta(seconds=end_time - start_time))
#print('Done! :) ')
