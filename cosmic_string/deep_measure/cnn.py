import matplotlib as mpl
mpl.use('agg')

import os
import ngene as ng
import numpy as np
import pylab as plt
from glob import glob
import tensorflow as tf
from utils import *
from sklearn.metrics import confusion_matrix
#from matplotlib.colors import LogNorm

print("\033[91m" + ' *cnn* :  10 classes of gmu  and between 1e-7 , 1e-3 including 0. loss func.: tf.losses.sigmoid_cross_entropy ' +"\033[0m")

"""
1 : tf.losses.sigmoid_cross_entropy , 20 classes: 1e-7 , 1e-3 , 3h training
2 : 

"""


#training_epochs = 100
#iterations=100
#n_s = 100
#learning_rate = 0.05
dofilt = 'n'

g_files = sorted(glob('../data/patches/ffp/*.npy'))
s_files = sorted(glob('../data/patches/s/*.npy'))

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
gmus = [0]+list(10**np.linspace(-7 , -3 , 9))
n_class = len(gmus)
dp = DataProvider(g_files,s_files,
                  gmus=gmus,
                  nx=100,ny=100,n_buffer=10,
                  reload_rate=10e5,filt=filt)

restore = os.path.isdir('./model/1')
restore = 0

def arch(x):
    return arch_maker(x,0,n_class)
       

def loss(y_true,x_out):
#    trsh = tf.constant(0.5,dtype=x_out.dtype,shape=tf.shape(x_out))
#    trsh = tf.fill(tf.shape(x_out), 0.5)
#    return tf.reduce_mean(tf.pow(y_true - x_out, 2))+0.1*tf.reduce_mean(1./(tf.abs(x_out-trsh)+0.5))

#    mean, var = tf.nn.moments(tf.reshape(x_out, [-1]), axes=[0])
#    return tf.reduce_mean(tf.pow(y_true - x_out, 2))+tf.abs(var-0.5)*1e-4/(var+1e-5)
#    return tf.losses.softmax_cross_entropy(y_true,x_out)

    return tf.losses.sigmoid_cross_entropy(y_true,x_out)    #loss1

#     mean, var = tf.nn.moments(tf.reshape(x_out, [-1]), axes=[0])
#    return tf.losses.sigmoid_cross_entropy(y_true,x_out) + 1e-4/(var+1e-5)

#     return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true,x_out))
     
#	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = x_out , labels = y_true))


conv = ng.Model(data_provider=dp,
                     optimizer=tf.train.AdamOptimizer,
                     loss = loss,
                     restore=restore,
                     model_add='./model/1',
                     arch=arch)

#for i in range(1):
#    print( 'set ' + str(i+1) + '/1')
#    conv.train(data_provider=dp,training_epochs=2, 
#                iterations=100, n_s=100,
#                learning_rate = 0.001, verbose=1)

#    for _ in range(5):
#        x,y = dp(1)
#        pred = conv.predict(x)
#        print(np.mean(x),np.argmax(y),np.argmax(pred))
#    	
#conv.train(data_provider=dp,training_epochs=2, 
#        iterations=100, n_s=100,
#        learning_rate = 0.001, verbose=1)

#conv.train(data_provider=dp,training_epochs=100000, 
#        iterations=100, n_s=100,
#        learning_rate = 0.009, verbose=1 , time_limit = 60)

#conv.train(data_provider=dp,training_epochs=100000, 
#        iterations=100, n_s=100,
#        learning_rate = 0.003, verbose=1 , time_limit = 60)
#        
#conv.train(data_provider=dp,training_epochs=100000, 
#        iterations=100, n_s=100,
#        learning_rate = 0.001, verbose=1 , time_limit = 60)

#Y  = [] 
#pr = []

#for _ in range(3000):
#    x,y = dp(1)
#    pred = conv.predict(x)
#    Y.append(np.argmax(y))
#    pr.append(np.argmax(pred))
#    if _%50 == 0:
#    	print(np.mean(x),np.argmax(y),np.argmax(pred))
##    plt.scatter(np.argmax(y),np.argmax(pred))
##    plt.savefig('0')

#conf = confusion_matrix(Y,pr)
#print(conf)
#np.save('conf1.npy' , conf)
#plt.imshow(confusion_matrix(Y,pr))
#plt.savefig('cm1')
 
gmus = [0]+list(10**np.linspace(-7 , -3 , 19))  	
conf = np.load('conf1.npy')  	
labels = []
for gmu in gmus:
    labels.append('{:3.2e}'.format(gmu))

fig= plt.figure(figsize=(40,40))
plt.imshow(conf)
plt.xticks(np.arange(20),labels , fontsize=30 , rotation = 45)
plt.yticks(np.arange(20),labels , fontsize = 30)
plt.title('Confusion Matrix' , fontsize=50)
plt.savefig('cm11')
    
#conv.train(data_provider=dp,training_epochs=training_epochs, 
#            iterations=iterations, n_s=n_s,
#            learning_rate = learning_rate, verbose=1)

#for _ in range(10):
#    x,y = dp(1)
#    pred = conv.predict(x)
#    print(y,pred)
