#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:27:23 2021

@author: busrayildiz
"""

# Representation of artificial neuron with a class in python

import numpy as np
 
class Neuron:
    
    def __init__(self, weights , activation, bias=0):
        self.weights = weights
        self.activation = activation
        self.bias = bias
        
    def output(self, x):
        total = np.dot(self.weights, x)
        return self.activation(total + self.bias)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

        
n = Neuron(np.array([3, 4, 5, 6]), sigmoid)   
result= n.output(np.array([3,4,5,6]))    
print(result)
         
        