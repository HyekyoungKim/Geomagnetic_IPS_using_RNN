# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:05:27 2019

@author: Hyekyoung
Reference: 
    [1] Indoor Positioning System using Geomagnetic Field with Recurrent Neural Network (Bae et al. 2018)
"""

import numpy as np
import pylab
import random
import math

# in meters
stepsize = 0.6
#interval = 0.6 # for grid

"""
Move in the specified direction
"""
def move(x_prev, y_prev, direction, stepsize, width, length):
    x = x_prev + stepsize * math.cos(math.radians(direction))
    y = y_prev + stepsize * math.sin(math.radians(direction))
    _continue = stepsize <= x <= (width - stepsize) and stepsize <= y <= (length - stepsize)
    return x, y, _continue

"""
Match each position [x,y] in source to the closest position [x,y] in target
"""
def match(source, target):
    source_num = source.shape[0]
    result = []
    for i in range(source_num):
        idx = ((target[:,:2] - source[i])**2).sum(1).argmin()
        result.append(target[idx])
    return np.array(result)


"""
Generate a trace within (width) * (length) testbed
"""
def generate_trace(full_data, total_steps, width, length):
    # creating two array for containing x and y coordinate
    # of size equals to the number of size and filled up with 0's
    x = np.zeros(total_steps + 1)
    y = np.zeros(total_steps + 1)
    
    # start position
    x[0] = random.random() * width
    y[0] = random.random() * length
#    print(x[0])
#    print(y[0])
    
    # filling the coordinates with random variables
    i = 1 # number of steps taken so far
    while i <= total_steps:
        if stepsize <= x[i - 1] <= (width - stepsize) \
        and stepsize <= y[i - 1] <= (length - stepsize):
            val = random.randrange(0, 360)
        elif x[i - 1] < stepsize and stepsize <= y[i - 1] <= (length - stepsize):
            val = random.randrange(-90, 91)
        elif x[i - 1] > (width - stepsize) \
        and stepsize <= y[i - 1] <= (length - stepsize):
            val = random.randrange(90, 271)
        elif stepsize <= x[i - 1] <= (width - stepsize) and y[i - 1] < stepsize:
            val  = random.randrange(0, 181)
        elif stepsize <= x[i - 1] <= (width - stepsize) \
        and y[i - 1] > (length - stepsize):
            val = random.randrange(180, 361)
        elif x[i - 1] < stepsize and y[i - 1] < stepsize:
            val = random.randrange(0, 91)
        elif x[i - 1] < stepsize and y[i - 1] > (length - stepsize):
            val = random.randrange(270, 361)
        elif x[i - 1] > (width - stepsize) and y[i - 1] < stepsize:
            val = random.randrange(90, 181)
        else:
            val = random.randrange(180, 271)
        steps = random.randint(1, math.ceil(math.sqrt(width ** 2 + length ** 2)))
        _continue = True
        for j in range(1, steps):
            if i <= total_steps and _continue:
                if random.choice([True, False]):
                    x[i], y[i], _continue = move(x[i - 1], y[i - 1], val, stepsize, width, length)
                else:
                    x[i], y[i] = x[i - 1], y[i - 1]
                i = i + 1
            else:
                break
    
    xy = np.array(list(zip(x, y)))
    #print(xy[:10,:])
#    print(xy[0])
    pylab.title("Random Walk ($n = " + str(total_steps) + "$ steps)")
    pylab.plot(xy[:,0].flatten(), xy[:,1].flatten(),'-o')
    pylab.savefig("walk/rand_walk" + str(total_steps) + ".png", bbox_inches="tight", dpi=600)
    pylab.show()
    
    return match(xy, full_data)
    #print(xy[:10,:2])
#    a = match(xy, full_data)
    # plotting stuff:
#    pylab.title("Random Walk Confined ($n = " + str(total_steps) + "$ steps)")
#    pylab.plot(a[:,0].flatten(), a[:,1].flatten())
#    pylab.savefig("rand_walk_confined" + str(total_steps) + ".png", bbox_inches="tight", dpi=600)
#    pylab.show()
#    
#    return xy
generate_trace(np.loadtxt('DataCollectionManualTagRound5.csv', delimiter=','),20,20.0,6.0)