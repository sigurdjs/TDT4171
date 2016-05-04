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

    def backPropagate(self,x,y,prnt=False):
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
        delta = self.cost_derivative(a_l[-1], y,prnt) * sigmoid_der(z_l[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,a_l[-2].T)
        for l in xrange(2,self.numOfLayers):
            z = z_l[-l] 
            s_der = sigmoid_der(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * s_der
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,a_l[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def updateWeights(self,nabla_b,nabla_w):
        self.weights = [w-(self.learningRate*nw) for w, nw in
                 zip(self.weights,nabla_w)]
        self.biases = [b-(self.learningRate*nb) for b, nb in
                 zip(self.biases,nabla_b)]
        return 0

    def updateWeightsAndPrint(self,nabla_b,nabla_w):
        self.weights = [w-(self.learningRate*nw) for w, nw in
                 zip(self.weights,nabla_w)]
        self.biases = [b-(self.learningRate*nb) for b, nb in
                 zip(self.biases,nabla_b)]
        print(self.weights)
        print(self.biases)
        print('this was one iteration')
        return 0
                    

    def cost_derivative(self, output_activations, y, prnt=False):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
	if prnt == True:
	    print(output_activations-y) 
        return (output_activations-y) 

def main():
    nn = NeuralNetwork([3,4,4,4,1],0.1)
    x1 = np.array([[0,0,0]])
    x2 = np.array([[0,1,0]])
    x3 = np.array([[0,0,1]])
    x4 = np.array([[1,1,0]])
    x5 = np.array([[0,1,1]])
    x6 = np.array([[1,0,0]])
    x7 = np.array([[1,0,1]])
    x8 = np.array([[1,1,1]])
    y1 = np.array([[0]])
    y2 = np.array([[0]])
    y3 = np.array([[0]])
    y4 = np.array([[1]])
    y5 = np.array([[1]])
    y6 = np.array([[0]])
    y7 = np.array([[1]])
    y8 = np.array([[0]])
    X = [x1,x2,x3,x4,x5,x6,x7,x8]
    y = [y1,y2,y3,y4,y5,y6,y7,y8]
    for i in range(0,500000):
        index = np.random.randint(8)
        if i%10000 == 0:
            print('progressing...')
            print(y[index])
            print(X[index])
	    nb,nw = nn.backPropagate(X[index].transpose(),y[index],True)
            nn.updateWeights(nb,nw)
        else:
	    nb,nw = nn.backPropagate(X[index].transpose(),y[index])
            nn.updateWeights(nb,nw)
#        nn.updateWeights(nb,nw)
#        nb,nw = nn.backPropagate(x2.transpose(),y2)
#        nn.updateWeights(nb,nw)
#        nb,nw = nn.backPropagate(x3.transpose(),y3)
#        nn.updateWeights(nb,nw)
#        nb,nw = nn.backPropagate(x4.transpose(),y4)
#        nn.updateWeights(nb,nw)

    test = np.array([[0,0,1]])
    test1 = np.array([[1,0,1]])
    test2 = np.array([[0,1,1]])
    test3 = np.array([[1,1,1]])
    print('tests')

    print(nn.propagate(test.T))
    print(nn.propagate(test1.T))
    print(nn.propagate(test2.T))
    print(nn.propagate(test3.T))


main()
