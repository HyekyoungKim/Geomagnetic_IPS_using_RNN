# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:28:00 2019

@author: Hyekyoung
References:
    [1] https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py
    [2] Indoor Positioning System using Geomagnetic Field with Recurrent Neural Network (Bae et al. 2018)
    [3] Geomagnetic Field Based Indoor Localization Using Recurrent Neural Networks (Jang et al. 2017)
"""

import sys
import time
import numpy as np
import tensorflow as tf
#import randomwalk as rw
from tensorflow.contrib import rnn

sys.stdout = open('output/training.txt','a')
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
width = 20.0 # x can be 0 ~ 20.0
length = 6.0 # y can be 0 ~ 6.0

# for RNN training
input_dim = 3 # magnetic field data: (vertical, horizontal, magnitude)
hidden_dim = 100 # output of RNN & input of FC
hidden_layer = 3  # the number of hidden layers of RNN
output_dim = 2 # position: (x, y)
train_p = 0.6 # proportion of data used for training
learning_rate = 0.001
training_epochs = 400
batch_size = 20 # size of each mini batch

traces_load = np.load('traces/map5_random_traces'+str(num_trace)+'.npz')
data_mag = traces_load['data_mag']
data_pos = traces_load['data_pos']
data_mag, mag_min, mag_max = MinMaxScaler(data_mag)
data_pos, pos_min, pos_max = MinMaxScaler(data_pos)

# split to train and testing
train_size = int(len(data_pos) * train_p)
test_size = len(data_pos) - train_size
train_mag = np.array(data_mag[0:train_size])
test_mag = np.array(data_mag[train_size:])
train_pos = np.array(data_pos[0:train_size])
test_pos = np.array(data_pos[train_size:])

# input placeholders
mag = tf.placeholder(tf.float64, [None, num_step + 1, input_dim])
pos = tf.placeholder(tf.float64, [None, num_step + 1, output_dim])

cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.relu) for _ in range(hidden_layer)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, mag, dtype=tf.float64)
pos_pred = tf.contrib.layers.fully_connected(
        outputs, output_dim, activation_fn=None)
        # We use the last cell's output#        
# cost/loss
loss = tf.reduce_mean(tf.square(pos_pred - pos)) # mean of the squares

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

save_file = 'saved/hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)\
+'_mini'+str(batch_size)+'_rate'+str(learning_rate)+'_epoch'+str(training_epochs)\
+'_step'+str(num_step)+'_trace'+str(num_trace)+'_p'+str(train_p)+'.ckpt'
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("==========================")
print("The number of hidden nodes: "+str(hidden_dim))
print("The number of hidden layers: "+str(hidden_layer))
print("The size of minibatches: "+str(batch_size))
print("Learning rate: "+str(learning_rate))
print("The number of training epochs: "+str(training_epochs))
print("The number of steps per trace: "+str(num_step))
print("The number of traces: "+str(num_trace))
print("Portion used to training: "+str(train_p))
loss_list = []
error_list = []
error_after_five_steps_list = []
training_begin_time = time.time()

for epoch in range(training_epochs):
    batch_count = int(train_mag.shape[0]/batch_size) # the number of batches
    for i in range(batch_count):
        batch_xs, batch_ys = train_mag[i*batch_size:i*batch_size+batch_size], train_pos[i*batch_size:i*batch_size+batch_size]
        _, l = sess.run([train, loss],
                        feed_dict={mag: batch_xs, pos: batch_ys})

    predict, ls = sess.run([pos_pred, loss], feed_dict={mag: train_mag, pos: train_pos})
    predict = InverseMinMaxScaler(predict, pos_min, pos_max)
    actual = InverseMinMaxScaler(train_pos, pos_min, pos_max)
    mean_error = np.mean(np.sqrt((np.square(actual-predict)).sum(axis=2)))
    mean_error_after_five_steps = np.mean(np.sqrt((np.square(actual[:,5:,:]-predict[:,5:,:])).sum(axis=2)))
    loss_list.append([epoch, ls])
    error_list.append([epoch, mean_error])
    error_after_five_steps_list.append([epoch, mean_error_after_five_steps])
    loss_array = np.array(loss_list)
    error_array = np.array(error_list)
    error_after_five_steps_array = np.array(error_after_five_steps_list)
    np.savez('result/error', loss=loss_array, error=error_array, error_five=error_after_five_steps_array)
    if (epoch + 1) % 10 == 0 :
        saver.save(sess, save_file, global_step=epoch+1)
        print("The time needed to train RNN ("+str(epoch+1)+" epochs): "+str(time.time() - training_begin_time))

#        print("Epoch", epoch, ", Batch", i, l)
training_end_time = time.time()
np.savetxt('loss/loss_hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)+'_mini'+str(batch_size)+'_rate'+str(learning_rate)+'_epoch'+str(training_epochs)+'_step'+str(num_step)+'_trace'+str(num_trace)+'.csv', loss_array, delimiter=',')
np.savetxt('loss/error_hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)+'_mini'+str(batch_size)+'_rate'+str(learning_rate)+'_epoch'+str(training_epochs)+'_step'+str(num_step)+'_trace'+str(num_trace)+'.csv', error_array, delimiter=',')
np.savetxt('loss/error_five_hnode'+str(hidden_dim)+'_hlayer'+str(hidden_layer)+'_mini'+str(batch_size)+'_rate'+str(learning_rate)+'_epoch'+str(training_epochs)+'_step'+str(num_step)+'_trace'+str(num_trace)+'.csv', error_after_five_steps_array, delimiter=',')


print("The time needed to train RNN: "+str(training_end_time - training_begin_time))
saver.save(sess, save_file)
