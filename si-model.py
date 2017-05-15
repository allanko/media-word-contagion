# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:32:45 2017

@author: Hannah
"""

import numpy as np
import matplotlib.pyplot as plt

SOURCES = 500

TIME_STEPS = 1000

A = np.load('adjacency.npy')

# S = np.zeros(((TIME_STEPS, SOURCES)))
# S[0] = [1,0,0,0,0,0,0,0,0,0]
# S[1] = [1,1,0,0,0,0,0,0,0,0]
# S[2] = [1,1,1,0,0,0,0,0,0,0]
# S[3] = [1,1,1,1,0,0,0,0,0,0]
# S[4] = [1,1,1,1,1,0,0,0,0,0]
# S[5] = [1,1,1,1,1,1,0,0,0,0]
# S[6] = [1,1,1,1,1,1,1,0,0,0]
# S[7] = [1,1,1,1,1,1,1,1,0,0]
# S[8] = [1,1,1,1,1,1,1,1,1,0]
# S[9] = [1,1,1,1,1,1,1,1,1,1]

S = np.load('states.npy')
print(S.shape)
#[::1000,:]
print(S.shape)

power = np.zeros((TIME_STEPS))

p = .5

#normalize by row ... WEIGHT A

A_weighted = A/np.sum(A, axis = 1).reshape((SOURCES,1))

A_importance = A*np.sum(A, axis=0)

A_importance_weighted = A_importance/np.sum(A_importance, axis = 1).reshape((SOURCES,1))


for t in range(TIME_STEPS-1):
    #print('actual state', S[t])
    prediction = p*np.dot(A_weighted, S[t])
    #print('prediction', prediction)
    probabilities = (1-S[t])*(np.abs(prediction - (1 - S[t+1])))
    power[t] = np.prod(probabilities[np.nonzero(probabilities)])
    #print('probability correct prediction', probabilities)

plt.plot(power)
plt.ylabel('Probability of Correct Prediction')
plt.show()
