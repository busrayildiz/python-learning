#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:17:40 2021

@author: busrayildiz
"""

"""
import numpy as np

dataset = np.loadtxt('test2.csv', delimiter=',', skiprows=1)

def standard_scaler(dataset):
    for col in range(dataset.shape[1]):
        mu = np.mean(dataset[:,col])
        std=np.std(dataset[:,col])
        dataset[:,col]=(dataset[:,col] - mu) / std
        
print(dataset, end='\n\n')
standard_scaler(dataset)
print(dataset)        
"""


#NORMALIZATION

import numpy as np

dataset = np.loadtxt('test2.csv', delimiter=',', skiprows=1)


def minmax_scaler(dataset):
    scaled_dataset= np.zeros_like(dataset)
    for col in range(dataset.shape[1]):
        #don't use min and max as a variable name because ther are built-in functions names !!
        minval= (dataset[:,col]).min()
        maxval= (dataset[:,col]).max()

        scaled_dataset[:, col] =(dataset[:, col] - minval) / (maxval - minval)
        
        return scaled_dataset
        
scaled_data = minmax_scaler(dataset)        
print(scaled_data)
        
        