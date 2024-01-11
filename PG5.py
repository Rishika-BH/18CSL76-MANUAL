# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:39:55 2024

@author: Rishi
"""

import numpy as np

inputNeurons=2
hiddenlayerNeurons=2
outputNeurons=2

input = np.random.randint(1,100,inputNeurons)
output = np.array([1.0,0.0])
hidden = np.random.rand(1,hiddenlayerNeurons)

hidden_bias=np.random.rand(1,hiddenlayerNeurons)
output_bias=np.random.rand(1,outputNeurons)
hidden_weights=np.random.rand(inputNeurons,hiddenlayerNeurons)
output_weights=np.random.rand(hiddenlayerNeurons,outputNeurons)

def sigmoid(layer):
    return i/(i + np.exp(-layer))

def gradient(layer):
    return layer*(i-layer)

for i in range(2000):
    hidden_layer = np.dot(input,hidden_weights)
    hidden_layer = sigmoid(hidden_layer+hidden_bias)
   
    output_layer = np.dot(hidden,output_weights)
    output_layer = sigmoid(output_layer+output_bias)
   
    error = (output-output_layer)
    gradient_outputLayer = gradient(output_layer)
   
    error_terms_output = gradient_outputLayer * error
   
    error_terms_hidden = gradient(hidden_layer) * np.dot(error_terms_output,output_weights.T)
   
    gradient_hidden_weights = np.dot(input.reshape(inputNeurons,1),error_terms_hidden.reshape(1,hiddenlayerNeurons))
    gradient_output_weights = np.dot(hidden.reshape(hiddenlayerNeurons,1),error_terms_output.reshape(1,outputNeurons))
   
    hidden_weights = hidden_weights + 0.05*gradient_hidden_weights
    output_weights = output_weights + 0.05*gradient_output_weights
   
    print("***************************************")
    print("Iteration:",i,"::::",error)
    print("#######Output#######",output_layer)