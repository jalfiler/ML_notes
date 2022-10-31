#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:41:48 2022

@author: jl
"""

# A Simple Neural Network With A Single Neuron

import numpy as np
from numpy import exp, array, random, dot

# Two features: W0 + W1x1 + W2x2 

# X1 and X2 = Input
# Out1...Out8 = Each Output


# Out1:
train_inputs = array([[0, 0], [0,1], [1, 0], [1, 1]])
train_outputs = array([[0, 0, 1, 1]]).T


random.seed(1)
synaptic_weights = 2 * random.random((2, 1)) - 1 #(2,1) two features

xrange = range

def train(train_inputs, train_outputs, iterations):
        for iteration in xrange(iterations):
 
            #getting output
            output = getoutput(train_inputs)

         	#calculating error
            error = train_outputs - output

            #calculating the adjustment
            adjustment = dot(train_inputs.T, error * sig_grad(output))
            global synaptic_weights
            synaptic_weights += adjustment


def getoutput(inputs):
    return sigmoid(dot(inputs, synaptic_weights))

def sigmoid(x):
	return 1 / (1 + exp(-x))

def sig_grad(x):
        return x * (1 - x)


print("Random starting synaptic weights - ")
print(synaptic_weights)


train(train_inputs, train_outputs, 10000)


print("synaptic weights after training: ")
print(synaptic_weights)

# Testing neural network 
print("For testing Out1 [0, 0, 1, 1]: ")
x= getoutput(array([0, 0, 1, 1]))
print(x)
