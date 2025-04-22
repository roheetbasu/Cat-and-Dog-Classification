import numpy as np

def Sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    Cache = Z # Useful for backpropagation
    
    return A, Cache

def relu(Z):
    A = np.maximum(0,Z)
    Cache = Z # useful for backpropagation
    
    return A, Cache

def relu_backward (dA, Cache):
    # dA post-activation Gradient 
    # Cache hold Z of present 
    # return 