#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:42:55 2022

@author: jl
"""

import numpy as np

# creating the input array
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print("Input:\n", X)

# shape of input array
print("\nShape of Input:", X.shape)


# converting the input in matrix form
X = X.T
print("Input in matrix form:\n", X)

# shape of input matrix
print("\nShape of Input Matrix:", X.shape)

# creating the output array
y = np.array([[0], [0], [1], [1]])

print("Actual Output:\n", y)

# output in matrix form
y = y.T

print("\nOutput in matrix form:\n", y)

# shape of input array
print("\nShape of Output:", y.shape)
