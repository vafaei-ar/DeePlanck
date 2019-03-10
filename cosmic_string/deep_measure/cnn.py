import matplotlib as mpl
mpl.use('agg')

import os
import ngene as ng
import numpy as np
import pylab as plt
#from matplotlib.colors import LogNorm
from glob import glob
from utils import *

print("\033[91m" + ' *cnn* :  cnn without any dropout , with kernel size = 3, filters =36, iters = 500 , Gu:[1e-5 , 9e-5]. applied on ffp10 simulations for large strings. learning_rate = 0.5 , data per iter = 1000' +"\033[0m")

training_epochs = 100
iterations=100
n_s = 100
learning_rate = 0.05
dofilt = 'n'

g_files = sorted(glob('../data/training_set/ffp10_p/*.npy'))
s_files = sorted(glob('../data/training_set/string_p/*.npy'))

if len(g_files)*len(s_files)==0:
    print('Somthing is wrong with initiation.')
    exit()

if dofilt[0]=='y':
    import ccgpack as ccg
    def filt(x):
        return ccg.filters(x,edd_method='sch')
else:
    filt = None

#omin,omax = np.log(self.gmb[0]),np.log(self.gmb[1])
#gmus = np.exp(np.random.uniform(omin,omax,n))
gmus = [0]+list(10**np.linspace(-5 , -1 , 4))
n_class = len(gmus)
dp = DataProvider(g_files,s_files,
                  gmus=gmus,
                  nx=10,ny=10,n_buffer=10,
                  reload_rate=10e5,filt=filt)

restore = os.path.isdir('./model')
restore = 0

def arch(x):
    return arch_maker(x,0,n_class)
    
    
import tensorflow as tf
def loss(y_true,x_out):
#    trsh = tf.constant(0.5,dtype=x_out.dtype,shape=tf.shape(x_out))
#    trsh = tf.fill(tf.shape(x_out), 0.5)
#    return tf.reduce_mean(tf.pow(y_true - x_out, 2))+0.1*tf.reduce_mean(1./(tf.abs(x_out-trsh)+0.5))

#    mean, var = tf.nn.moments(tf.reshape(x_out, [-1]), axes=[0])
#    return tf.reduce_mean(tf.pow(y_true - x_out, 2))+tf.abs(var-0.5)*1e-4/(var+1e-5)
#    return tf.losses.softmax_cross_entropy(y_true,x_out)
    return tf.losses.sigmoid_cross_entropy(y_true,x_out)


    
conv = ng.Model(data_provider=dp,
                     optimizer=tf.train.AdamOptimizer,
                     loss = loss,
                     restore=restore,
                     model_add='./model',
                     arch=arch)

conv.train(data_provider=dp,training_epochs=10, 
            iterations=100, n_s=100,
            learning_rate = 0.1, verbose=1)

for _ in range(5):
    x,y = dp(1)
    pred = conv.predict(x)
    print(np.mean(x),np.argmax(y),np.argmax(pred))
    
conv.train(data_provider=dp,training_epochs=10, 
        iterations=100, n_s=100,
        learning_rate = 0.01, verbose=1)

for _ in range(20):
    x,y = dp(1)
    pred = conv.predict(x)
    print(np.mean(x),np.argmax(y),np.argmax(pred))
#conv.train(data_provider=dp,training_epochs=training_epochs, 
#            iterations=iterations, n_s=n_s,
#            learning_rate = learning_rate, verbose=1)

#for _ in range(10):
#    x,y = dp(1)
#    pred = conv.predict(x)
#    print(y,pred)
