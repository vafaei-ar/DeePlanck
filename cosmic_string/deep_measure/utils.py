import numpy as np
import tensorflow as tf
from random import choice,shuffle

#g_ps = []
#s_ps = []

#for g in glob.glob('../data/training_set/ffp10_p/*.npy'):
#    g_ps.append(np.load(g))
#g_ps = np.array(g_ps)
#print 'g_ps.shape:',g_ps.shape

#for s in glob.glob('../data/training_set/string_p/*.npy'):
#    s_ps.append(np.load(s))
#s_ps = np.array(s_ps)
#print 's_ps.shape:' , s_ps.shape

#def dp(n):
#    l  = 200
#    x = []
#    y = []
#    for i in range(n):
#	    gm = np.random.uniform(10,1000)
#	    gm = gm* 1e-7
#	    
#	    r1 = np.random.randint(0,len(g_ps))       
#	    ri,rj = np.random.randint(0,2048-l,2)
#	    r2 = np.random.randint(0,len(s_ps))
#	    ri,rj = np.random.randint(0,2048-l,2)
#	    	    
#	    gp = g_ps[r1][ri:ri+l,rj:rj+l]
#	    sp = s_ps[r2][ri:ri+l,rj:rj+l]

#	    x.append(gp+gm*sp)
#	    y.append(-np.log(gm))

#    return np.expand_dims(np.array(x) , -1) , np.expand_dims(np.array(y) , -1)

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
        n_class = len(gmus)
#        x,y = self.get_data()
        X = []
        Y = []
        for i in range(n):                
            n,s = self.get_data()
            inds = np.arange(n_class)
            shuffle(inds)
            gmu = gmus[inds[0]]
#            gmu = choice(gmus)
            sn = self.pre_process(n,s,gmu)
#            rand = np.random.randint(0,2)
#            sn = sn-sn+rand
            X.append(sn)
#            Y.append(-np.log10(gmu+1e-30))
            lbl = n_class*[0]
            lbl[inds[0]] = 1
            Y.append(lbl)
            
        X = np.array(X)
        Y = np.array(Y)
 
#    def __call__(self, n, gmus=None): 
#        rp = np.random.uniform(0,1)
#    
#        if gmus is None:
#            gmus = self.gmus
##        x,y = self.get_data()
#        X = []
#        Y = []
#        for i in range(n):                
#            n,s = self.get_data()
#            gmu = choice(gmus)
#            sn = self.pre_process(n,s,gmu)
#            rand = int(np.random.uniform(0,1)>rp)
#            sn = sn-sn+rand
#            X.append(sn)
##            Y.append(-np.log10(gmu+1e-30))
#            lbl = [0,0]
#            lbl[rand] = 1
#            Y.append(lbl)
#            
#        X = np.array(X)
#        Y = np.array(Y)
    
        return X,Y

#x,y = dp_total(7)
#fig,ax=plt.subplots(1,1,figsize=(5,5))
#ax.imshow(x[0,:,:,0],norm=LogNorm(),cmap=plt.get_cmap('jet'))
#plt.title('G + Gu*S')
#plt.savefig('x_lognorm ')
#fig,ax=plt.subplots(1,1,figsize=(5,5))
#ax.imshow(x[0,:,:,0])
#plt.title('G + Gu*S')
#plt.savefig('x')
#print(x.shape,y.shape)
#exit()
#l = 200
#nx,ny,n_channel = l,l,1
 
def arch_t(x_in):    

    print("\033[91m ============== Begin ============== \033[0m")

    x1 = tf.layers.conv2d(x_in,filters=36,kernel_size=3,
    strides=(1, 1),padding='same',activation=tf.nn.relu)
    print(x1)

    x2 = tf.layers.average_pooling2d(x1,pool_size=2,strides=2)
    print(x2)

    x2 = tf.layers.conv2d(x2,filters=36,kernel_size=3,
    strides=(2, 2),padding='same',activation=tf.nn.relu)
    print(x2)
    
    x3 = tf.layers.average_pooling2d(x2,pool_size=2,strides=2)
    print(x3)

    x3 = tf.layers.conv2d(x3,filters=36,kernel_size=3,
    strides=(2, 2),padding='same',activation=tf.nn.relu)
    print(x3)
    x4 = tf.layers.average_pooling2d(x3,pool_size=3,strides=2)
    print(x4)


    x4 = tf.layers.conv2d(x4,filters=36,kernel_size=3,strides=(2, 2),padding='same',
            activation=tf.nn.relu)
    print(x4)
    #x5 = tf.layers.average_pooling2d(x4,pool_size=2,strides=2)

    x5 = tf.layers.conv2d(x4,filters=36,kernel_size=3,strides=(2, 2),padding='same',
            activation=tf.nn.relu)
    print(x5)
    #x5 = tf.layers.average_pooling2d(x5,pool_size=2,strides=2)

    #x5 = tf.layers.conv2d(x5,filters=36,kernel_size=3,strides=(2, 2),padding='same',
            #activation=tf.nn.relu)

    #x5 = tf.layers.average_pooling2d(x5,pool_size=2,strides=2)


    x7 = tf.contrib.layers.flatten(x5)
    x7 = tf.nn.dropout( x7, keep_prob=0.6)
    print(x7)
    x7 = tf.layers.dense(x7, 10 , activation=tf.nn.relu)
    print(x7)
    y_out = tf.layers.dense(x7, 1 , activation=tf.nn.relu)
    print(y_out)
    
    print("\033[91m =============== END =============== \033[0m")
    return y_out
    
def arch_maker(x,n_conv,n_class):
    #x_in = tf.placeholder(tf.float32,[None,nx,ny,n_channel])
    #y_true = tf.placeholder(tf.float32,[None , n_channel])
    #learning_rate = tf.placeholder(tf.float32)
    print("\033[91m ============== Begin ============== \033[0m")
#    for _ in range(n_conv):
#        x = tf.layers.conv2d(x,filters=16,kernel_size=5,
#                              strides=(1, 1),padding='same')
#        print(x)
#        x = tf.layers.batch_normalization(x)
#        print(x)
#        x = tf.nn.relu(x)
#        print(x)
    for _ in range(n_conv):
        x = tf.layers.conv2d(x,filters=4,kernel_size=3,
                              strides=(1, 1),padding='same')
        print(x)
        x = tf.layers.batch_normalization(x)
        print(x)
        x = tf.nn.relu(x)
        print(x)
        x = tf.layers.average_pooling2d(x,pool_size=2,strides=2)
        print(x)

    x = tf.contrib.layers.flatten(x)
    print(x)
    x = tf.nn.dropout( x, keep_prob=0.6)
    print(x)
    x = tf.layers.dense(x, 20 , activation=tf.nn.relu)
    print(x)
    y = tf.layers.dense(x, n_class, activation=tf.nn.softmax)
    print(y)
    print("\033[91m =============== END =============== \033[0m")
    return y
    
