#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:08:19 2022

@author: jl
"""

# Neural Networks with MLPClassification

import random


X = []
Y = []
Length = 20
TrainPortion = 0.75
TrainCut = int(Length*TrainPortion)

for i in range(Length):
    tmp1 = random.uniform(-10,+10)
    tmp2 = random.uniform(-10,+10)
    X.append([tmp1,tmp2])
    if (tmp1*tmp2)>=0:
        Y.append(1)
    else:
        Y.append(0)

         
import numpy as np

Xtrain = np.array(X[:TrainCut])
Ytrain = np.array(Y[:TrainCut])
Xtest = np.array(X[TrainCut:])
Ytest = np.array(Y[TrainCut:])


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier( hidden_layer_sizes=(2), random_state=1, max_iter=2000)
clf.fit(Xtrain, Ytrain)
TrainEstimate = clf.predict(Xtrain)
avgErrorTrain = sum(abs(TrainEstimate-Ytrain))/len(Ytrain)
print("Average Train Error:", avgErrorTrain)
Yestimate = clf.predict(Xtest)
avgError = sum(abs(Yestimate-Ytest))/len(Ytest)
print("Average Test Error:", avgError)


# Plot
import matplotlib.pyplot as plt

xaxis = np.random.normal(0, 5, 1000)   #concentrated at 0 and have gapping of 5 in 1000 random dots
yaxis = np.random.normal(0, 5, 1000)    
plt.scatter(xaxis, yaxis)                   
plt.show()       

#-------------------------
'''
1.) Test error with no hidden layer: (hidden_layer_sizes=(1)
Average Train Error: 0.2
Average Test Error: 0.4 

2.) Test error with one hidden layer and two neurons: (hidden_layer_sizes=(1,1)
Average Train Error: 0.4666666666666667
Average Test Error: 0.4

'''
                                 
