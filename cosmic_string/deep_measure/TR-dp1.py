import matplotlib as mpl
mpl.use('agg')

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pylab as plt

def dp (n , n_class):
    
    X = []
    Yc= []
    Yr= []


    for i in range(n):

        x = np.random.normal(0,1,(20,20))
        rand = (n_class) * np.random.rand()
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        rand3 = np.random.rand()
        x[1,11] = np.sin(rand)-rand1
        x[18,2] = rand1-rand2
        x[5,19] = rand2-np.sin(rand)
        X.append(x)

        rand2 = int(rand)
        yc = np.zeros((n_class))
        yc[rand2] += 1
        Yc.append(yc)

        Yr.append(rand)
    Yr = np.expand_dims(Yr, -1)
    X = np.expand_dims(X , -1)

    return np.array(X) , np.array(Yc) ,np.array(Yr) 

#for _ in range(5):
#    x , yc , yr = dp(1,4)
#    assert x[:,0,0,:]==yr,'error'
#    print (yc)
    
#_,yc,_ = dp(1000,4)
#yc.mean(axis=0)
#print(x,yc)

n_class = 5
n_conv  = 3
conv_trainable = True

x_in = tf.placeholder(tf.float32,[None,20,20,1])    
learning_rate = tf.placeholder(tf.float32)
conv_trainable = tf.Variable(True)
drop_out = tf.placeholder(tf.float32)
y_true_c = tf.placeholder(tf.float32,[None , n_class])
y_true_r = tf.placeholder(tf.float32,[None , 1])

print("\033[91m =============== Begin =============== \033[0m")

for i in range(n_conv):
	x = tf.layers.conv2d(x_in,filters=4,kernel_size=3,strides=(2, 2),padding='same',
	                        activation=tf.nn.relu , trainable= conv_trainable)
	print(x)
	x = tf.layers.average_pooling2d(x,pool_size=2,strides=2)
	print(x)

if n_conv == 0 :
	x = x_in

x_f = tf.contrib.layers.flatten(x)
print(x_f)

print("\033[91m =============== Classification... =============== \033[0m")

x = tf.layers.dense(x_f, 10 , activation=tf.nn.relu , trainable= conv_trainable)
print(x)
x = tf.layers.dropout(x, rate=drop_out , training=conv_trainable)
print(x)
x_c = tf.layers.dense(x, n_class , activation=tf.nn.softmax , trainable=conv_trainable)
print(x_c)

print("\033[91m =============== Regression... =============== \033[0m")

x = tf.layers.dense(x_f, 10 , activation=tf.nn.relu , trainable=True)
print(x)
x = tf.layers.dropout(x, rate=drop_out , training=True)
print(x)
x_r = tf.layers.dense(x, 1 , activation=tf.nn.relu , trainable= True)
print(x_r)

print("\033[91m =============== END =============== \033[0m")

cost1 = tf.losses.huber_loss(y_true_c,x_c)
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(cost1)

cost2 = tf.losses.mean_squared_error(y_true_r,x_r)
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)

sess = tf.InteractiveSession()
saver=tf.train.Saver()

init = tf.global_variables_initializer()

try:
    saver.restore(sessd, './model/trl1')
except:
    sess.run(init)


print("\033[94m =============== cl training... =============== \033[0m")
for j in range(10):
	for i in range(2000):
#for j in range(0):
#	for i in range(1):

		x, yc , yr = dp(100,n_class)
		        
		_,c = sess.run([optimizer1, cost1], feed_dict=
		                 {x_in: x, y_true_c: yc , y_true_r: yr, learning_rate: 0.001 , drop_out : 0.4})

	print (j, ':' ,c)

Y = []
pr= []
for _ in range(3000):
	x, yc , yr = dp(1,4)
	pred = sess.run(x_c, feed_dict={x_in: x, drop_out : 0.0})
	Y.append(np.argmax(yc))
	pr.append(np.argmax(pred))
conf = confusion_matrix(Y,pr)
print(conf)
plt.imshow(confusion_matrix(Y,pr),origin='lower')

#plt.close()


accu = np.trace(conf)/3000
print(accu)

saver.save(sess, './model/trl1')

assign_op = conv_trainable.assign(False)
sess.run(assign_op)

aa = sess.run(conv_trainable)
print(aa)

print("\033[94m =============== reg training... =============== \033[0m")
for j in range(10):
	
	for i in range(2000):

		x , yc , yr = dp(100,n_class)
		        
		_,c = sess.run([optimizer2, cost2], feed_dict=
		                 {x_in: x, y_true_c: yc, y_true_r: yr , learning_rate: 0.001 , drop_out : 0.4})

	print (j, ':' ,c)

Y = []
pr= []
for _ in range(300):
	x, yc , yr = dp(1,4)
	pred = sess.run(x_r, feed_dict={x_in: x, drop_out : 0.0})[0,0]
	Y.append(yr[0,0])
	pr.append(pred)
plt.scatter(Y,pr)
plt.plot(Y,Y,'r')
plt.title('2st approach')
plt.savefig('cm1.jpg')





exit()

# ### to compare this approach with only regression , set all trainable options True

n_class = 4
n_conv  = 2
conv_trainable = True

x_in = tf.placeholder(tf.float32,[None,20,20,1])    
learning_rate = tf.placeholder(tf.float32)
drop_out = tf.placeholder(tf.float32)
y_true_c = tf.placeholder(tf.float32,[None , n_class])
y_true_r = tf.placeholder(tf.float32,[None , 1])

print("\033[91m =============== Begin =============== \033[0m")

for i in range(n_conv):
	x = tf.layers.conv2d(x_in,filters=4,kernel_size=3,strides=(2, 2),padding='same',
	                        activation=tf.nn.relu , trainable= conv_trainable)
	print(x)
	x = tf.layers.average_pooling2d(x,pool_size=2,strides=2)
	print(x)

if n_conv == 0 :
	x = x_in

x_f = tf.contrib.layers.flatten(x)
print(x_f)
x_f = tf.layers.dropout(x_f, rate=0.6 , training=True)
print(x_f)

print("\033[91m =============== Classification... =============== \033[0m")

x = tf.layers.dense(x_f, 10 , activation=tf.nn.relu , trainable= conv_trainable)
print(x)
x_c = tf.layers.dense(x, n_class , activation=tf.nn.softmax , trainable=conv_trainable)
print(x_c)

print("\033[91m =============== Regression... =============== \033[0m")

x = tf.layers.dense(x_f, 10 , activation=tf.nn.relu , trainable=True)
print(x)
x_r = tf.layers.dense(x, 1 , activation=tf.nn.relu , trainable= True)
print(x_r)

print("\033[91m =============== END =============== \033[0m")


# In[13]:


cost2 = tf.losses.mean_squared_error(y_true_r,x_r)
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)

sess = tf.InteractiveSession()
saver=tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)


# In[14]:


print("\033[94m =============== reg training... =============== \033[0m")
for j in range(20):
	
	for i in range(1000):

		x , yc , yr = dp(100,4)
		        
		_,c = sess.run([optimizer2, cost2], feed_dict=
		                 {x_in: x, y_true_c: yc, y_true_r: yr , learning_rate: 0.001 , drop_out : 0.6})

	print (j, ':' ,c)


# In[15]:


Y = []
pr= []
for _ in range(300):
	x, yc , yr = dp(1,4)
	pred = sess.run(x_r, feed_dict={x_in: x})[0,0]
	Y.append(yr[0,0])
	pr.append(pred)
plt.scatter(Y,pr)
plt.plot(Y,Y,'r')
plt.title('2nd approach')
plt.savefig('2')



