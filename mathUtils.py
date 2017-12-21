import numpy as np

def sigmoid(x):
    # Equivalent to 1/(1 + math.e**(-x))
    return 1.0/(1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))
