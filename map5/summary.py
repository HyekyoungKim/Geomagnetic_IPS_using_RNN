# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:21:29 2019

@author: Samsung
"""

import numpy as np

load = np.load('result/error.npz')
loss = load['loss']
error = load['error']
error_five = load['error_five']

np.savetxt('result/loss.csv', loss, delimiter=',')
np.savetxt('result/error.csv', error, delimiter=',')
np.savetxt('result/error_five.csv', error_five, delimiter=',')

