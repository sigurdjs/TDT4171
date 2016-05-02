import numpy as np

def sigmoid(x):
    return 1.0/(1.0 - np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(sigmoid(x)-1.0)


def random_float(low,high,size): #Returns an array of size  with random numbers 
    return ((high-low)*np.random.rand(1,size) + low) 

    
class NeuralNetwork:
    def __init__(self,numInputs,numOutputs,numHidden,learningRate = 0.001):
        #Initialize input weights
        r = np.zeros(numHidden,numInputs)
        for i in range(0,numHidden):
            r[i] = random_float(-1.0,1.0,numInputs)
        self.weights.append(r)
        #Initialize hidden weights
        
        #Initialize output weights
        self.weights.append(random_float(-1.0,1.0,numHidden))
        self.learningRate = learningRate
        
        def propagate(X,bias):
           a[0] = sigmoid(np.dot(self.weights[0],X) + bias)
           
           #Iterate hidden layers if more than one
           output = sigmoid(np.dot(self.weights[1],a[0]) + bias) 
            
            
