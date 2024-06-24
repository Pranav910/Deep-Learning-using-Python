# importing necessary modules

import numpy as np
import pandas as pd

# loading dataset using pandas

dataset = pd.read_csv('Neural Network from scratch/PlacementDataset.csv')
X, y = dataset[['IQ', 'CGPA']], dataset[['Placement']]

# splitting the dataset into test and train splits

X_test, X_train, y_test, y_train = X[:800].to_numpy(), X[800:].to_numpy(), y[:800].to_numpy(), y[800:].to_numpy()

# Defining the neural network class

class NeuralNetwork:

    def __init__(self):
        self.totalInputParameters = 0
        self.layers = []
        self.totalParameters = 0
        self.totalWeights = 0
        self.totalBiases = 0
        self.parameters = {}

    def add(self, Layer):
        self.layers.append(Layer)
        if Layer.type == 'input':
            self.totalInputParameters = Layer.nodes*Layer.wsize + 1
            self.totalParameters += Layer.nodes*Layer.wsize
        elif Layer.type == 'hidden' or Layer.type == 'output':
            self.totalInputParameters = self.layers[0].nodes*self.layers[0].wsize + self.layers[1].nodes
            self.totalParameters += Layer.nodes*Layer.wsize + Layer.nodes

    def summary(self):
        print("===================================================")
        print("|| Summary                                       ||")
        print("===================================================")
        print(f"|| Total Trainable Parameters : {self.totalParameters}               ||")
        print("===================================================")
        print(f"|| Total weights : {self.totalWeights}                            ||")
        print("===================================================")
        print(f"|| Total biases : {self.totalBiases - self.layers[0].biases.size}                              ||")
        print("===================================================")
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def softmax(self, z):
        expZ = np.exp(z)
        return expZ/(np.sum(expZ, 0))

    def relu(self, z):
        return np.maximum(0, z)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def derivative_relu(self, z):
        return z > 0
    
    def derivative_tanh(self, z):
        return (1 - np.power(z, 2))
        

    def feedForward(self, X):
        
        forward_cache = {}
        parameters = self.parameters
        forward_cache['A0'] = X.reshape(2, 1)
        L = len(self.layers) - 1
        
        for i in range(1, L):
            forward_cache['Z' + str(i)] = parameters['W' + str(i)].dot(forward_cache['A' + str(i - 1)]) + parameters['B' + str(i)]
            forward_cache['A' + str(i)] = self.relu(forward_cache['Z' + str(i)])
            
        forward_cache['Z' + str(L)] = parameters['W' + str(L)].dot(forward_cache['A' + str(L - 1)]) + parameters['B' + str(L)]
        forward_cache['A' + str(L)] = self.sigmoid(forward_cache['Z' + str(L)])
        
        print(forward_cache)
            
        return forward_cache['A' + str(L)], forward_cache
        
        
    def computeCost(self, obs, exp, X):

        m = X.size
        
        cost = -(1/m) * np.sum(exp*np.log(obs) + (1 - exp)*np.log(1 - obs))
        return np.squeeze(cost)
    
    def backProp(self, obs, exp, parameters, forward_cache):
        
        grads = {}
        L = len(self.layers) - 1
        m = self.layers[0].nodes
        
        
        grads['dZ' + str(L)] = obs - exp
        grads['dW' + str(L)] = (1/m) * np.dot(grads['dZ' + str(L)], forward_cache['A' + str(L - 1)].T)
        grads['dB' + str(L)] = (1/m) * np.sum(grads['dZ' + str(L)], axis=1, keepdims=True)
        
        for i in reversed(range(1, L)):
            grads['dZ' + str(i)] = np.dot(parameters['W' + str(i + 1)].T, grads['dZ' + str(i + 1)]) * self.derivative_relu(forward_cache['A' + str(L)])
            grads['dW' + str(i)] = (1/m) * np.dot(grads['dZ' + str(i)], forward_cache['A' + str(i - 1)].T)
            grads['dB' + str(i)] = (1/m) * np.sum(grads['dZ' + str(i)], axis=1, keepdims=True)
            
        return grads
            
    def updateParameters(self, grads, parameters, lr):
        
        L = len(self.layers)
        
        for i in range(1, L):
            
            parameters['W' + str(i)] = parameters['W' + str(i)] - lr * grads['dW' + str(i)]
            parameters['B' + str(i)] = parameters['B' + str(i)] - lr * grads['dB' + str(i)]

        return parameters
            
        
    def compile(self):
        
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)] = np.random.rand(self.layers[i].nodes, self.layers[i - 1].nodes)
            self.parameters['B' + str(i)] = np.zeros((self.layers[i].nodes, 1))
            
        print(self.parameters)
            
    def fit(self, X, y):
        
        parameters = self.parameters
        cost = 0
        
        for i in range(len(X)):
            
            
            obs, forward_cache = self.feedForward(X[i])
            cost = self.computeCost(obs, y[i], X)
            grads = self.backProp(obs, y[i], parameters, forward_cache)
            parameters = self.updateParameters(grads, parameters, 0.001)
        

        return cost
    
    def predict(self, X):
        
        result, fc = self.feedForward(X)
        return result


class Layer:
    
    def __init__(self, nodes, wsize, type):
        self.nodes = nodes
        self.wsize = wsize
        self.type = type

model = NeuralNetwork()
model.add(Layer(2, 2, type='input'))
model.add(Layer(2, 1, type='hidden'))
model.add(Layer(3, 2, type='hidden'))
model.add(Layer(1, 0, type='output'))
model.compile()
model.fit(X_train, y_train)
result = model.predict(np.array([129,2.548110441374412]))
print(result)