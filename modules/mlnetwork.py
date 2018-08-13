#Basic multi-layer Perceptron#
#Based on https://github.com/FlorianMuellerklein/Machine-#

import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1.0 - y) 

def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist
 
def tanh(x):
    return np.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y

class MLP_Network(object):
    def __init__(self, input, hidden, output, iterations = 50, rate = 0.01, 
                l2_in = 0, l2_out = 0, momentum = 0, rate_decay = 0, 
                output_activation = 'logistic', verbose = True, debug = False):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param l2: L2 regularization term
        :param momentum: momentum
        :param rate_decay: how much to decrease learning rate by on each iteration (epoch)
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to print error rates while training
        """
        #setup params
        self.iterations = iterations
        self.rate = rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = 0
        self.output_activation = output_activation
        self.verbose = verbose
        self.debug = debug
        
        #setup arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output
        
        # setup array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)
        
        # create randomized weights 
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
        
        
    def feedForward(self, inputs):
        """
        :param inputs: input data
        :return activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs!')
            
        #input activation
        self.ai[0:self.input-1] = inputs
        
        #hidden activation
        sum = np.dot(self.wi.T, self.ai)
        self.ah = tanh(sum)
        
        if(self.debug and self.n == self.maxN):
            print("Inputs",self.ai[0:10])
            print("Hiddens sum", sum)
            print("Hiddens sample",self.ah[0:10])
        #output activations
        sum = np.dot(self.wo.T, self.ah)         
        if self.output_activation == 'logistic':
            self.ao = sigmoid(sum)
        elif self.output_activation == 'softmax':
            self.ao = softmax(sum)
        else:
            raise ValueError('Incompatible output layer activation function')
            
        if(self.debug and self.n == self.maxN):
            print("Output weights sample", self.wo[0:10,0])
            print("output sum", sum)
            print("Output after full iteration", self.ao)
            
        return self.ao
    
    def backPropagate(self, targets):
        """
        :param targets: y values
        :return: error
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets!')
            
        if(self.debug and self.n == self.maxN):
            print("Targets",targets)
            
        #calculate error terms for output
        if self.output_activation == 'logistic':
            output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        elif self.output_activation == 'softmax':
           output_deltas = -(targets - self.ao)
        else:
            raise ValueError('Incompatible output layer activation function')
        
        if(self.debug and self.n == self.maxN):
            print("Output deltas",output_deltas)
            
        #calculate error terms for hidden
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dtanh(self.ah) * error
        
        if(self.debug and self.n == self.maxN):
            print("Hidden deltas",hidden_deltas)
            
        #update the weights connecting hidden to outputs
        delta = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        regularized = self.l2_out * self.wo
        
        self.wo -= self.rate * (delta + regularized) + self.co * self.momentum
        self.co = delta
        
        #update the weights connecting input to hidden
        delta = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0],1))
        regularized = self.l2_in * self.wi
        self.wi -= self.rate * (delta + regularized) + self.ci * self.momentum
        self.ci = delta
        
        if(self.debug and self.n == self.maxN):
            print("Updated input weight sample", self.wi[0,0:10])
            print("Updated output weight sample", self.wo[0:10,0])
        
        #calculate error
        error = 0.0
        if self.output_activation == 'logistic':
            error = sum(0.5 * (targets - self.ao)**2)
        elif self.output_activation == 'softmax':
            error = -sum(targets * np.log(self.ao))
            
        return error
    
    def train(self, patterns):
        if self.verbose == True:
            if self.output_activation == 'logistic':
                print('Using logistic sigmoid activation')
            elif self.output_activation == 'softmax':
                print('Using softmax activation')
        
        num_example = np.shape(patterns)[0]
        
        for i in range(self.iterations):
            print("Iteration: ",i)
            error = 0.0
            random.shuffle(patterns)
            self.n = 0
            self.maxN = len(patterns) - 1
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
                #end after setup + first feedforward + backpropogate
                if(self.debug):
                    self.n += 1
            if i % 10 == 0 and self.verbose == True:
                error = error/num_example
                print('Training error %-.5f' % error)
            self.rate = self.rate * (self.rate / (self.rate + (self.rate * self.rate_decay)))
    
    def predict(self, x):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in x:
            predictions.append(self.feedForward(p))
        return predictions
    
    def test(self, patterns):
        """
        output targets vs predictions
        """
        nCorrect = 0
        nTotal = 0
        for p in patterns:
            nTotal += 1
            target = p[1].index(1)
            guess = self.feedForward(p[0])
            guessVal = np.argmax(guess)
            print(target, '->', guessVal)
            if(target == guessVal):
                nCorrect += 1
        accuracy = nCorrect/nTotal * 100
        print('Accuracy:', accuracy)


    
                
        