# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:23:24 2019

@author: Hyekyoung
"""

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

sys.stdout = open('output/new_test.txt','a')

tf.reset_default_graph()

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return (numerator / (denominator + 1e-7)), np.min(data, 0), np.max(data, 0)

def InverseMinMaxScaler(data, min_array, max_array):
    return min_array + data * (max_array - min_array + 1e-7)


num_step = 20 # number of steps per trace
num_trace = 10000 # number of traces
width = 20.0 # x can be 0 ~ 3.0
length = 6.0 # y can be 0 ~ 9.6

# for RNN training
input_dim = 3 # magnetic field data: (vertical, horizontal, magnitude)
hidden_dim = 10 # output of RNN & input of FC
hidden_layer = 1 # the number of hidden layers of RNN
output_dim = 2 # position: (x, y)
train_p = 0.6 # proportion of data used for training
validation_p = 0.2 #proportion of data used for validation
learning_rate = 0.001
training_epochs = 400
checkpoint = 400
batch_size = 20 # size of each mini batch
first_n = 5
    
traces_load = np.load('traces/map5_random_traces'+str(num_trace)+'.npz')
data_mag = traces_load['data_mag']
data_pos = traces_load['data_pos']
data_mag_scale, mag_min, mag_max = MinMaxScaler(data_mag)
data_pos_scale, pos_min, pos_max = MinMaxScaler(data_pos)

# split to train and testing
train_size = int(len(data_pos_scale) * train_p)
validation_size = int(len(data_pos_scale) * validation_p)
test_mag = np.array(data_mag_scale[train_size + validation_size : ])
test_pos = np.array(data_pos_scale[train_size + validation_size : ])

# input placeholders
mag = tf.placeholder(tf.float64, [None, num_step + 1, input_dim])
pos = tf.placeholder(tf.float64, [None, num_step + 1, output_dim])

cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.relu) \
                         for _ in range(hidden_layer)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, mag, dtype=tf.float64)
pos_pred = tf.contrib.layers.fully_connected(
        outputs, output_dim, activation_fn=None)
        # We use the last cell's output#        
# cost/loss
loss = tf.reduce_mean(tf.square(pos_pred - pos)) # mean of the squares

save_file = 'saved/hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)\
+'_mini'+str(batch_size)+'_rate'+str(learning_rate)+'_epoch'+str(training_epochs)\
+'_step'+str(num_step)+'_trace'+str(num_trace)+'_p'+str(train_p)+'.ckpt-'+str(checkpoint)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, save_file)

test_predict, l = sess.run([pos_pred, loss], feed_dict={mag: test_mag, pos: test_pos})

test_pos = InverseMinMaxScaler(test_pos, pos_min, pos_max)
test_predict = InverseMinMaxScaler(test_predict, pos_min, pos_max)

mean_error = np.mean(np.sqrt((np.square(test_pos-test_predict)).sum(axis=2)))
step_error = np.mean(np.sqrt((np.square(test_pos-test_predict)).sum(axis=2)), axis=0)
np.savetxt('result/step_error.csv', step_error, delimiter=',')
mean_error_after_five_steps = np.mean(np.sqrt((np.square(test_pos[:,5:,:]-test_predict[:,5:,:])).sum(axis=2)))
max_error = np.max(np.sqrt((np.square(test_pos-test_predict)).sum(axis=2)))
max_error_after_five_steps = np.max(np.sqrt((np.square(test_pos[:,5:,:]-test_predict[:,5:,:])).sum(axis=2)))
min_error = np.min(np.sqrt((np.square(test_pos-test_predict)).sum(axis=2)))
min_error_after_five_steps = np.min(np.sqrt((np.square(test_pos[:,5:,:]-test_predict[:,5:,:])).sum(axis=2)))
print("The number of hidden nodes: "+str(hidden_dim))
print("The number of hidden layers: "+str(hidden_layer))
print("The size of minibatches: "+str(batch_size))
print("The learning rate: "+str(learning_rate))
print("The number of training total epochs: "+str(training_epochs))
print("The number of training epochs so far: "+str(checkpoint))
print("---------------------------------")
print("Mean squared error: "+str(l))
print("Mean error: "+str(mean_error)+" m")
print("Max error: "+str(max_error)+" m")
print("Min error: "+str(min_error)+" m")
print("Mean error after first 5 steps: "+str(mean_error_after_five_steps)+" m")
print("Max error after first 5 steps: "+str(max_error_after_five_steps)+" m")
print("Min error after first 5 steps: "+str(min_error_after_five_steps)+" m")
print("=================================")
plt.figure(figsize=(9.6, 4.8))
#plt.title(str(num_step)+' steps, '\
#          +str(train_p*100)+'% of '+str(num_trace)+' traces')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
j=2
plt.plot(test_pos[j,:,0].flatten(), test_pos[j,:,1].flatten(), '-o', label='actual path')
plt.plot(test_predict[j,:,0].flatten(), test_predict[j,:,1].flatten(), '-o', label='predicted path')
#for i in range(num_step+1):
#    plt.text(test_pos[j,i,0], test_pos[j,i,1], str(i), fontsize=9)
#    plt.text(test_predict[j,i,0], test_predict[j,i,1], str(i), fontsize=9)
plt.text(test_pos[j,0,0]+0.2, test_pos[j,0,1], "start", fontsize=11)
plt.text(test_predict[j,0,0]+0.1, test_predict[j,0,1]-0.3, "start", fontsize=11)
plt.legend(loc='lower left')
plt.xlim(7, 22)
plt.ylim(0, 6)
plt.savefig('figure/validation_hnode'+str(hidden_dim)+'hlayer'+str(hidden_layer)\
            +'bsize'+str(batch_size)+'lrate'+str(int(learning_rate*10000))\
            +'epoch'+str(training_epochs)+'trace'+str(num_trace)+'.png')
