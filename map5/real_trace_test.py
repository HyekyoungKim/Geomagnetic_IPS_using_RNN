# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:08:00 2019

@author: Hyekyoung
"""

import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

sys.stdout = open('output/real_trace_test.txt', 'a')
tf.reset_default_graph()

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return (numerator / (denominator + 1e-7)), np.min(data, 0), np.max(data, 0)

def MinMaxScalerForPosition(data, min_array, max_array):
    numerator = data - min_array
    denominator = max_array - min_array
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def InverseMinMaxScaler(data, min_array, max_array):
    return min_array + data * (max_array - min_array + 1e-7)

num_step = 50
input_dim = 3 # magnetic field data: (vertical, horizontal, magnitude)
hidden_dim = 100 # output of RNN & input of FC
hidden_layer = 1 # the number of hidden layers of RNN
output_dim = 2 # position: (x, y)
training_epochs = 400
checkpoint = 400 
num_trace = 10000

# x, y, vertical, horizontal, magnitude
full_data = np.loadtxt('DataCollectionManualTagTrace.csv', delimiter=',')

data_mag = full_data[:,2:]
data_pos = full_data[:,:2]

pos_min = np.array([0, 0])
pos_max = np.array([20, 6])
data_mag_scale, mag_min, mag_max = MinMaxScaler(data_mag)
data_pos_scale = MinMaxScalerForPosition(data_pos, pos_min, pos_max)

# input placeholders
mag = tf.placeholder(tf.float64, [None, num_step, input_dim])
pos = tf.placeholder(tf.float64, [None, num_step, output_dim])

cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.relu) \
                         for _ in range(hidden_layer)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, mag, dtype=tf.float64)
pos_pred = tf.contrib.layers.fully_connected(
        outputs, output_dim, activation_fn=None)
        # We use the last cell's output#        
# cost/loss
loss = tf.reduce_mean(tf.square(pos_pred - pos)) # mean of the squares

save_file = 'saved/hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)\
+'_mini20_rate0.001_epoch'+str(training_epochs)+'_step20_trace'+str(num_trace)\
+'_p0.6.ckpt-'+str(checkpoint)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, save_file)

begin_time = time.time()
test_predict, l = sess.run([pos_pred, loss], 
                           feed_dict={mag: [data_mag_scale], pos: [data_pos_scale]})
end_time = time.time()
test_predict = InverseMinMaxScaler(test_predict, pos_min, pos_max)

mean_error = np.mean(np.sqrt((np.square(data_pos-test_predict)).sum(axis=2)))
mean_error_after_five = np.mean(np.sqrt((np.square([data_pos[5:,:]]-test_predict[:,5:,:])).sum(axis=2)))
print("The number of hidden nodes: "+str(hidden_dim))
print("The number of hidden layers: "+str(hidden_layer))
print("The number of random traces: "+str(num_trace))
print("The number of training epochs: "+str(checkpoint))
print("The time needed to predict a trace: "+str(end_time - begin_time))
print("---------------------------------")
print("Mean squared error: "+str(l))
print("Mean error: "+str(mean_error)+" m")
print("Mean error after first five steps: "+str(mean_error_after_five)+" m")
print("=================================")
plt.figure(figsize=(13.2, 4.8))
plt.title('50 steps')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(data_pos[:,0].flatten(), data_pos[:,1].flatten(), '-o', label='actual path')
plt.plot(test_predict[:,:,0].flatten(), test_predict[:,:,1].flatten(), '-o', label='predicted path')
for i in range(num_step):
	plt.text(data_pos[i,0]+0.1, data_pos[i,1], str(i), fontsize=9)
	plt.text(test_predict[0,i,0]+0.1, test_predict[0,i,1], str(i), fontsize=9)
plt.legend(loc='lower left')
plt.xlim(-1, 21)
plt.ylim(-1, 7)
plt.savefig('figure/real_hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)+'_bsize20_lrate001_epoch'+str(checkpoint)+'.png')
