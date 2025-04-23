import numpy as np

def Sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    Cache = Z # Useful for backpropagation
    
    return A, Cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    Cache = Z # useful for backpropagation
    
    return A, Cache

def relu_backward (dA, Cache):
    # dA post-activation Gradient 
    # Cache hold Z of present 
    # return dZ gradient of the cost wrt to Z
    Z = Cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert(dZ.shape == Z.shape)
    
    return dZ

def Sigmoid_backward(dA, Cache):
    # dA post- activation Gradient
    # Cache hold Z of present
    # return dZ gradient of the cost wrt to Z
    Z = Cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA* s * (1-s)
    assert(dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters(n_x, n_h, n_y):
    """ 
     ---Argument---
     n_x : Size of input layer
     n_h : Size of hidden layer
     n_y : Size of output layer
     
     ---returns---
     
     parameters : Dictationary containing parameter
     
                    W1 : Shape of (n_h,n_x)
                    b1 : Shape of (n_h,1)
                    W2 : Shape of (n_y,n_h)
                    b2 : Shape of (n_y,1)
    
    """
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h,n_x))
    assert(W2.shape == (n_y,n_h))
    assert(b1.shape == (n_h,1))
    assert(b2.shape == (n_y,1))
    
    parameters = {
        "W1" : W1,
        "W2" : W2,
        "b1" : b1,
        "b2" : b2      
    }

    return parameters

def initialize_parameters_deep(layers_dim):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dim)  # Number of layers (including input/output)
    
    # Initialize parameters for layers 1 to L-1
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dim[l], layers_dim[l-1]) * np.sqrt(2 / layers_dim[l-1])
        parameters[f'b{l}'] = np.zeros((layers_dim[l], 1))
    return parameters

def linear_forward(A,W,b):
    
    Z =W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_forward_activation(A_prev,W,b,activation):
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if (activation == "Sigmoid"):
        A, activation_cache = Sigmoid(Z)
        
    elif (activation == "relu" ):
        A, activation_cache = relu(Z)
        
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # Number of layers with parameters
    
    # Loop through all hidden layers (ReLU activation)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(
            A_prev, parameters[f'W{l}'], parameters[f'b{l}'], activation="relu"
        )
        caches.append(cache)
    
    # Output layer (sigmoid activation)
    AL, cache = linear_forward_activation(
        A, parameters[f'W{L}'], parameters[f'b{L}'], activation="Sigmoid"
    )
    caches.append(cache)
    
    return AL, caches
    
    
def compute_cost(AL, Y):
    
    m = Y.shape[1] # numbers of inputs
    AL = np.clip(AL, 1e-5, 1 - 1e-5)
    # compute loss using AL and Y
    cost = (-1./m) * (np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T))
    
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ) 
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_backward_activation(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "Sigmoid":
        dZ = Sigmoid_backward(dA,activation_cache)
        dA, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA, dW, db = linear_backward(dZ,linear_cache)
        
    return dA, dW, db

def L_model_backward(AL, Y, Caches):
    # return grads dA, dW, db
    
    L = len(Caches) 
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = Caches[L-1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_backward_activation(dAL, current_cache, activation='Sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = Caches[l]
        dA_temp, dW_temp, db_temp = linear_backward_activation(grads[f'dA{l+1}'], current_cache, activation='relu')
        grads[f'dA{l}'] = dA_temp
        grads[f'dW{l+1}'] = dW_temp
        grads[f'db{l+1}'] = db_temp
        
        
    return grads
        
def update_parameter(parameter, grads, learning_rates):
    
    L = len(parameter) // 2 
    
    for l in range(L):
        parameter[f'W{l+1}'] = parameter[f'W{l+1}'] - learning_rates * grads[f'dW{l+1}']
        parameter[f'b{l+1}'] = parameter[f'b{l+1}'] - learning_rates * grads[f'db{l+1}']
        
    return parameter

def predict(X, Y, parameters):
    
    m = Y.shape[1]
    # n = len(parameters) // 2 
    p = np.zeros((1,m))
    
    # forward propagation
    probas, cache = L_model_forward(X, parameters)
    
    p = (probas >= 0.5).astype(int)   
        
    return p 


        
    