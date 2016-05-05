import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def random_float(low,high,size): #Returns an array of size  with random numbers 
    return ((high-low)*np.random.rand(1,size) + low) 

class NeuralNetwork:    
    def __init__(self,sizes,learningRate = 0.001):
        self.numOfLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.learningRate = learningRate
#        print(self.biases)
#        print(self.weights)
        
    def propagate(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def backPropagate(self,x,err,prnt=False):
        a = x
        a_l = [x]
        z_l = [] 
        delta_l = []
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,a) + b
	    z_l.append(z)
            a = sigmoid(z)
            a_l.append(a)
        delta = err * sigmoid_der(z_l[-1])
        nabla_b[-1] = 0
        nabla_w[-1] = np.dot(delta,a_l[-2].T)
        for l in xrange(2,self.numOfLayers):
            z = z_l[-l] 
            s_der = sigmoid_der(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * s_der
            nabla_b[-l] = 0
            nabla_w[-l] = np.dot(delta,a_l[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def updateWeights(self,nabla_b,nabla_w):
        self.weights = [w-(self.learningRate*nw) for w, nw in
                 zip(self.weights,nabla_w)]
        self.biases = [b-(self.learningRate*nb) for b, nb in
                 zip(self.biases,nabla_b)]

    def countMisorderedPairs(self,pairs):
        correct = 0
        incorrect = 0
        f_a, f_b = self.reshape(pairs)
        for a,b in zip(f_a,f_b):
            o_a = self.propagate(a)
            o_b = self.propagate(b)
            if o_a > o_b:
                correct += 1
            else:
                incorrect += 1
        return (correct,incorrect)

    def reshape(self,pairs):
        f_a = []
        f_b = []
        for pair in pairs:
            features_a = np.asarray(pair[0].features)
            features_b = np.asarray(pair[1].features)
            f_a.append(features_a.reshape(len(features_a),1))
            f_b.append(features_b.reshape(len(features_b),1))
        return (f_a,f_b)

    def train(self,pairs):
        f_a, f_b = self.reshape(pairs)
        for a,b in zip(f_a,f_b):
            o_a = self.propagate(a)
            o_b = self.propagate(b)
            err = (1-sigmoid(o_a-o_b))
            nw_a, _ = self.backPropagate(a,err)
            nw_b, _ = self.backPropagate(b,err)
            nw = nw_a - nw_b
            updateWeights(0,nw)

    def cost_derivative(self, output_activations, y, prnt=False):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
	if prnt == True:
	    print(output_activations-y) 
        return (output_activations-y) 
