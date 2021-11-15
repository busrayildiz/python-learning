#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:17:40 2021

@author: busrayildiz
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
        
        