# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:24:24 2019

@author: Hyekyoung
"""

import sys
import numpy as np
import time
import randomwalk as rw

num_step = 50 # number of steps per trace
num_trace = 10 # number of traces
width = 20.0 # x can be 0 ~ 20.0
length = 6.0 # y can be 0 ~ 6.0
map_num = 5

sys.stdout = open('output/generating_map'+str(map_num)+'_random_trace'+str(num_trace)+'.txt','a')

# x, y, vertical, horizontal, magnitude
full_data = np.loadtxt('DataCollectionManualTagRound'+str(map_num)+'.csv', delimiter=',')

data_mag = []
data_pos = []
trace_begin_time = time.time()
for i in range(num_trace):
    trace = rw.generate_trace(full_data, num_step, width, length)
    _mag = trace[:,2:]
    _pos = trace[:,:2]
    data_mag.append(_mag)
    data_pos.append(_pos)

trace_end_time = time.time()
print("The time needed to generate random traces: "+str(trace_end_time - trace_begin_time)+" s")
np.savez('traces/map'+str(map_num)+'_random_traces'+str(num_trace), data_mag=data_mag, data_pos=data_pos)
