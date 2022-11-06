#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 20:53:12 2022

@author: jl
"""

import csv

# Data import
glob = dict()       
with open('covid19_global_dataset.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile)
    for r in datareader:
        c = r[0]
        if c=='week':
            weekID = r[1:]
        else:
            tmp = []
            for i in range(0,21):
                tmp.append(0)
            darray = r[1:]
            for i in range(0,len(darray)):
                t = int(weekID[i])
                d = int(darray[i])
                if t<21:
                    tmp[t] += d
            glob[c] = tmp    
            

# New Cases
allNews = []
for c in glob:
    tmp = glob[c]
    tmp2 = [tmp[0]]
    allNews.append(tmp[0])
    for i in range(1,len(tmp)):
        tmp2.append(tmp[i] - tmp[i-1])
        allNews.append(tmp[i] - tmp[i-1])
    glob[c] = tmp2


# Build Dataset
X = []
Y = []
step = 15
for c in glob:
    tmp = glob[c]
    for j in range(0,len(tmp)-step-1):
        stest = sum(tmp[j:j+step])
        if stest>0:
            X.append(tmp[j:j+step])
            Y.append(tmp[j+step])
            

import numpy as np

npX = np.array(X)
npY = np.array(Y)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(npX, npY, test_size=0.33)


# Use MLPRegressor for linear regression

from sklearn.neural_network import MLPRegressor            

clf = MLPRegressor(hidden_layer_sizes=(10,10),random_state=1,max_iter=2000)
clf.fit(Xtrain,Ytrain)
Yestimate = clf.predict(Xtest)
print("score = ",clf.score(Xtest,Ytest))
err1 = (sum((Yestimate-Ytest)**2)/len(Ytest))**0.5
print("RMS error = ",err1)


# K-fold cross validation

from sklearn.model_selection import KFold

# define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# ------------------------------------------------
'''
10-fold cv with 3 different parameters: 5,7,15

step = 5
score =  0.8984868389680073
RMS error =  16422.77397265614

step = 7 
score =  0.9459923551319138
RMS error =  15109.155966967883

step = 15
score =  0.9276301439856532
RMS error =  43673.87364269177

'''

