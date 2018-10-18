#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


# In[7]:


np.random.seed(1234)
tf.set_random_seed(1254)
IMAGE_HEIGHT = 28
IMAGE_WIDTH  = 28
NUM_CLASSES = 10
l2_lamb = 1e-5
lr = 1e-3
BATCH_SIZE = 2048
NUM_BATCHES = 500000
D = 198800
intrinsic_dims = [200,300,400,450,500,550,600,650,675,700,725,750]
#intrinsic_dims = [800]


# In[8]:


class Data(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        randos = np.random.choice(60000,60000, replace=False)
        validation_randos = randos[:12000]
        train_randos = randos[12000:]
        self.val_x, self.val_y = self.reshape(x_train[validation_randos]), y_train[validation_randos]
        self.x_train, self.y_train = self.reshape(x_train[train_randos]), y_train[train_randos]
        self.x_test = self.reshape(x_test)
        self.y_train = keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        self.y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        self.val_y = keras.utils.to_categorical(self.val_y,NUM_CLASSES)
    def reshape(self,x):
        return np.reshape(x, [-1,IMAGE_WIDTH,IMAGE_HEIGHT,1])    
    def get_batch(self):
        choices = np.random.choice(len(self.x_train), size=BATCH_SIZE)
        val_choices = np.random.choice(len(self.val_x), size=BATCH_SIZE)
        return self.x_train[choices], self.y_train[choices], self.val_x[val_choices], self.val_y[val_choices]


# In[14]:


def create_p(D, intrinsic_dim):
    x = np.random.normal(0,1,size=(D, intrinsic_dim))
    return tf.constant(x, dtype=tf.float32)


# In[5]:


def get_weights(theta_D,shape,starting_point):
    dim = shape[0] * shape[1]
    ending_point = starting_point+dim
    vals = theta_D[starting_point:ending_point,]
    return tf.reshape(vals,shape),ending_point


# In[6]:


def dense(x,W,soft=False):
    W_prime = tf.matmul(x,W)
    if soft:
        return tf.nn.softmax(W_prime)
    return tf.nn.relu(W_prime)


# In[7]:


def run(intrinsic_dim,lr):
    fo = open("results_{}.txt".format(intrinsic_dim), "w")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    p = create_p(D,intrinsic_dim)
    theta_not = tf.get_variable("theta_not",[D, 1], tf.float32,tf.random_normal_initializer())
    theta_d = tf.get_variable("theta_d", [intrinsic_dim, 1], tf.float32,tf.zeros_initializer())
    theta_D = theta_not + tf.matmul(p,theta_d)
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH,IMAGE_HEIGHT,1])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
    input_layer = tf.reshape(x, [-1, 28*28])
    W1, end1 = get_weights(theta_D,(784,200),0)
    dense1 = dense(input_layer,W1)
    W2,end2 = get_weights(theta_D, (200,200), end1)
    dense2 = dense(dense1, W2)
    W3,_ = get_weights(theta_D,(200,10), end2)
    logits = dense(dense2,W3,soft=True)
    loss = tf.keras.backend.categorical_crossentropy(target=y, output=logits)
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1), 
                                      predictions=tf.argmax(logits,1))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, var_list=theta_d)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    sess.run([init,init2])
    num_step = NUM_BATCHES
    for i in range(1,3):
        num_step= int(num_step/i**3)
        lr = lr/(10**(i-1))
        for i in range(0, num_step):
            x_np, y_np,val_x, val_y = data.get_batch()
            loss_np, _, acc = sess.run([loss, train_op, acc_op], feed_dict={x: x_np, y: y_np, learning_rate:lr})
            if i %200== 0 :
                val_loss, val_acc = sess.run([loss, acc_op], feed_dict={x: val_x, y: val_y})
                fo.write("LR:{}:{}:Val {} \n".format(lr,i,val_acc))
                print("LR:{}:{}:Val {} \n".format(lr,i,val_acc))
    fo.close()


# In[9]:


data = Data()


# In[9]:


for x in intrinsic_dims:
   tf.reset_default_graph()
   run(x,lr)
print("Done!")


# In[ ]:


print(acc)

