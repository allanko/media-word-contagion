# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:32:45 2017

@author: Hannah
"""

import numpy as np
import matplotlib.pyplot as plt
import math

SOURCES = 500

TIME_STEPS = 168

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

S = np.load('states_clintonspeech.npy')
#plt.plot(np.sum(S, axis=1))

#normalize by row ... WEIGHT A

A_weighted = A/np.sum(A, axis = 1).reshape((SOURCES,1))

A_importance = A*np.sum(A, axis=0)

A_importance_weighted = A_importance/np.sum(A_importance, axis = 1).reshape((SOURCES,1))

A_realsi = A/np.sum(A, axis=0)
sums = np.sum(A,axis=0)
print(sums[sums==1])
A_realsi[np.isnan(A_realsi)] = 0
print(1 - A_realsi)
A_realsi = np.log(1 - A_realsi)
print(A_realsi)

p_exps = 40

final_score = np.zeros((p_exps))
final_score_rand = np.zeros((p_exps))
final_score_si = np.zeros((p_exps))

for e in range(1, p_exps):
    power = np.zeros((TIME_STEPS))
    power_rand = np.zeros((TIME_STEPS))
    power_si = np.zeros((TIME_STEPS))
    p = 0.001 * e

    # try removing sources with no articles?

    for t in range(1, TIME_STEPS-2):
        #print('actual state', S[t])

        prediction = p*(np.dot(A_weighted, S[t]))

        dot_result = np.dot(A_realsi, S[t])
        dot_result[np.isnan(dot_result)] = float('-inf')
        prediction_si = p * (1 - np.exp(dot_result))

        #print(np.sum(prediction_si) / len(S[t]))
        prediction_rand = p * np.ones((len(S[t])))
        #print('prediction', prediction)
        probabilities = (1-S[t])*(np.abs(prediction - (1 - S[t+1])))
        probabilities_rand = (1-S[t])*(np.abs(prediction_rand - (1 - S[t+1])))
        probabilities_si = (1-S[t])*(np.abs(prediction_si - (1 - S[t+1])))
        #print(probabilities)
        power[t] = np.sum(np.log(probabilities[np.nonzero(probabilities)]))
        power_rand[t] = np.sum(np.log(probabilities_rand[np.nonzero(probabilities_rand)]))
        power_si[t] = np.sum(np.log(probabilities_si[np.nonzero(probabilities_si)]))
        #print(power[t])
        #print('probability correct prediction', probabilities)
    final_score[e] = np.sum(power[1:-2])
    final_score_rand[e] = np.sum(power_rand[1:-2])
    final_score_si[e] = np.sum(power_si[1:-2])

plt.plot(final_score[1:])
plt.plot(final_score_rand[1:])
plt.plot(final_score_si[1:])
print(np.max(final_score[1:]))
print(np.max(final_score_rand[1:]))
print(np.max(final_score_si[1:]))
plt.ylabel('Probability of Correct Prediction')
plt.show()
