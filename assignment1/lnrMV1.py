#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:58:03 2018

The Goal is to model Wine Quality against the rest of 11 attributes



@author: nikhila
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#Read File

RD = pd.read_csv('/home/nikhila/assignment1/winequality-white.csv',sep = ';')

Y = RD.quality
X = RD.drop(['quality'],axis = 1)
for i in X.columns:
    print(i)
    print(X[i].mean())
    print(X[i].std())
Z = X.T
"""
for i in X.shape[0]:
    print(i)
    
Compute derivative of Cost function

theta = np.zeros((X-train.shape[1]),1)
h_theta = np.zeros((X-train.shape[0],1))

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        h_theta[i,0] + = theta[j]*X_train.iloc[i,j]
X_train.shape[0] = 3918
X_train.shape[1] = 11
function to calc sum
for each example
for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        X_train.iloc[i,j]*h_theta[i] - Y_train[i]*X_train.iloc[i,j]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""