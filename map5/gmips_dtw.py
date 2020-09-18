# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:24:07 2019

@author: Hyekyoung
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import random
import randomwalk as rw

sys.stdout = open('output/dtw_test.txt','a')

num_step = 20 # number of steps per trace
num_trace = 10000 # number of traces
width = 3.0 # x can be 0 ~ 3.0
length = 9.6 # y can be 0 ~ 9.6
total_steps = 20000

# for DTW testing
target_p = 0.9 # proportion of data included in the target pattern set

# x, y, vertical, horizontal, magnitude
full_data = np.loadtxt('DataCollectionManualTagRound5.csv', delimiter=',')

#dataMag = []
#dataPos = []

#for i in range(1, num_trace):
#    trace = rw.generate_trace(full_data, num_step, width, length)
#    _mag = trace[:,2:]
#    _pos = trace[:,:2]
#    #print(_mag, "->", _pos)
#    dataMag.append(_mag)
#    dataPos.append(_pos)
full_trace = rw.generate_trace(full_data, total_steps, width, length)
dataMag = full_trace[:, 2:]
dataPos = full_trace[:, :2]

error = []
_time = []
for i in range(10):
#    steps = random.randrange(3, 11)
    steps = 21
    test_trace = rw.generate_trace(full_data, steps, width, length)
    _pos = test_trace[:, :2]
    _mag = test_trace[:, 2:]
    start_time = time.time()
    min_dist, _ = fastdtw(_mag, dataMag[:steps], dist=euclidean)
    match = 0
    for j in range(1, total_steps - steps + 1):
        dist, _ = fastdtw(_mag, dataMag[j:j+steps], dist=euclidean)
        if dist < min_dist:
            match = j
    finish_time = time.time()
    print(str(steps)+" steps")
    print(str(finish_time-start_time)+" sec.")
    _error = np.sqrt((np.square(_pos[-1] - dataPos[match+steps])).sum())
    error.append(_error)
    _time.append(finish_time-start_time)
    print("Error: "+str(_error)+"\n")
    
# split to pattern set and testing set
#target_size = int(len(dataPos) * target_p)
#test_size = len(dataPos) - target_size
#targetMag = np.array(dataMag[0:target_size])
#testMag = np.array(dataMag[target_size:])
#targetPos = np.array(dataPos[0:target_size])
#testPos = np.array(dataPos[target_size:])
#
#pos_pred = []
#for i in range(1):
#    min_dist, _ = fastdtw(testMag[i], targetMag[0], dist=euclidean)
#    match = 0
#    for j in range(1, target_size):
#        dist, _ = fastdtw(testMag[i], targetMag[j], dist=euclidean)
#        if dist < min_dist:
#            match = j
#    pos_pred.append(targetPos[j])

error = np.array(error)
mean_error = np.mean(error) 
_time = np.array(_time)
mean_time = np.mean(_time)
print("Mean error: "+str(mean_error)+" m")
print("Mean time: "+str(mean_time)+" s")
print("==================================")

#plt.xlabel('x')
#plt.ylabel('y')
#plt.plot(testPos[-1,:,0].flatten(), testPos[-1,:,1].flatten(), label='actual path')
#plt.plot(pos_pred[-1,:,0].flatten(), pos_pred[-1,:,1].flatten(), label='predicted path')
#plt.legend(loc='lower left')
#plt.show()