# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:32:45 2017

@author: Hannah
"""

import numpy as np
import matplotlib.pyplot as plt


SOURCES = 500
TIME_STEPS = 300
Arand = np.random.randint(2, size=(SOURCES, SOURCES))
Aones = np.ones((SOURCES, SOURCES))
S = np.zeros(((TIME_STEPS, SOURCES)))
ll = np.zeros((TIME_STEPS-1))


A_w = np.ones((SOURCES, SOURCES))
A_iw = np.ones((SOURCES, SOURCES))
# a couple A matrices to use
def generate_A(unweightedA):
    A_W = unweightedA/np.sum(unweightedA, axis = 1).reshape((SOURCES,1))
    A_importance = unweightedA*np.sum(unweightedA, axis=0)
    A_IW = A_importance/np.sum(A_importance, axis = 1).reshape((SOURCES,1))
    
#generates test time data
def generate_S(weightedA, p):
    S[0][0]=1
    for t in range(TIME_STEPS-1):
        prediction = p*np.dot(weightedA, S[t])
        newS = (np.random.uniform(size = SOURCES)<prediction)*1
        indToChange = np.nonzero(1-S[t])
        S[t+1] = np.ones(SOURCES)
        S[t+1][indToChange] = newS[indToChange]

def generate_S_rand(weightedA, p):
    S[0][0]=1
    for t in range(TIME_STEPS-1):
        newS = (np.random.uniform(size = SOURCES)<p)*1
        indToChange = np.nonzero(1-S[t])
        S[t+1] = np.ones(SOURCES)
        S[t+1][indToChange] = newS[indToChange]
               
def predictivity_score_rand(weightedA, someS, p):
    prediction_success = np.zeros((SOURCES))
    for t in range(someS.shape[0]-1):
        prediction_success = (1-someS[t])*(np.abs(p - (1 - someS[t+1])))
        ll[t] = np.sum(np.log((prediction_success[np.nonzero(prediction_success)])))
    plt.plot(ll)
    plt.ylabel('log loss')
    plt.show()
    
def predictivity_score(weightedA, someS, p):
    prediction_success = np.zeros((SOURCES))
    for t in range(someS.shape[0]-1):
        prediction = p*np.dot(weightedA, someS[t])
        prediction_success = (1-someS[t])*(np.abs(prediction - (1 - someS[t+1])))
        ll[t] = np.sum(np.log((prediction_success[np.nonzero(prediction_success)])))
    plt.plot(ll)
    plt.ylabel('log loss')
    plt.show()
 
#generate_A(Aones)
generate_S_rand(A_iw, .02)
predictivity_score_rand(A_w, S, .7)
plt.plot(np.sum(S, axis = 1))
plt.ylabel('accumulation of S')
plt.show()
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#normalize by row ... WEIGHT A

 
#    print('prediction', prediction)
#    probabilities = (1-S[t])*(np.abs(prediction - (1 - S[t+1])))
#    #power[t] = np.prod(probabilities[np.nonzero(probabilities)])
#    power[t] = np.sum(np.log((probabilities[np.nonzero(probabilities)])))
#    probabilities_rand = (1-S[t])*(np.abs(p_rand - (1 - S[t+1])))
#    power_rand[t] = np.sum(np.log((probabilities_rand[np.nonzero(probabilities_rand)])))
#    print('probability correct prediction', probabilities)
    
#plt.plot(power)
#plt.plot(power_rand)
#plt.ylabel('Probability of Correct Prediction')
#plt.show()


 
