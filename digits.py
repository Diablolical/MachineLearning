# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:55:37 2018

@author: Austin
"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from mlnetwork import MLP_Network
import matplotlib.pyplot as plt
import time
import random


def plot_digit(X, idx):
    img = X[idx].reshape(8,8)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.show()

def test_load():
    X  = load_digits().data
    plot_digit(X,0)
    y = load_digits().target
    print(X.shape)
    print(y.shape)
    
def load_digit_data():
    digits = load_digits()
    data = digits.data
    y = digits.target
    data = scale(data)
    out = []
    for i in range(data.shape[0]):
        targets = [0]*10
        targets[y[i]] = 1
        tupleData = list((data[i,:].tolist(), targets))
        out.append(tupleData)
        
    return out

def run():
     
    start_time = time.time()
    
    x = load_digit_data()
    
    random.shuffle(x)
    
    test = x[0:200]
    train = x[200:]

    input_len = len(x[0][0])
    output_len = len( x[0][1])
    
    NN = MLP_Network(input_len, 2000, 10, iterations = 50, rate = 0.1, momentum = 0.1, rate_decay = 0.01)
    
    NN.train(train)
     
    #NN.test(x)
    
    end_time = time.time()
    
    print("Completed training in ",end_time - start_time, "seconds")
    
    NN.test(test)
    
if __name__ == '__main__':
    run()
    #test_load()