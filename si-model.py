# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:32:45 2017

@author: Hannah
"""

import numpy as np
import matplotlib.pyplot as plt

SOURCES = 500

TIME_STEPS = int(20000 / 24)

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

S = np.load('states.npy')[::24,:]
#plt.plot(np.sum(S, axis=1))

power = np.zeros((TIME_STEPS))
power_rand = np.zeros((TIME_STEPS))
p = 0.1

#normalize by row ... WEIGHT A

A_weighted = A/np.sum(A, axis = 1).reshape((SOURCES,1))

A_importance = A*np.sum(A, axis=0)

A_importance_weighted = A_importance/np.sum(A_importance, axis = 1).reshape((SOURCES,1))

A_realsi = A/np.sum(A, axis=0)
A_realsi[np.isnan(A_realsi)] = 0

for t in range(1, TIME_STEPS-2):
    #print('actual state', S[t])
    prediction = p*(1 - np.dot(A_realsi, S[t]))
    #print(np.sum(prediction) / len(S[t]))
    prediction_rand = p * np.ones((len(S[t])))
    #print('prediction', prediction)
    probabilities = (1-S[t])*(np.abs(prediction - (1 - S[t+1])))
    probabilities_rand = (1-S[t])*(np.abs(prediction_rand - (1 - S[t+1])))
    #print(probabilities)
    power[t] = np.sum(np.log(probabilities[np.nonzero(probabilities)]))
    power_rand[t] = np.sum(np.log(probabilities_rand[np.nonzero(probabilities_rand)]))
    #print(power[t])
    #print('probability correct prediction', probabilities)

#plt.plot(power_rand[1:-2])
plt.plot(power[1:-2])
print(np.sum(power[1:-2]))
plt.ylabel('Probability of Correct Prediction')
plt.show()
