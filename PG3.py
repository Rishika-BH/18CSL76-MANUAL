# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:23:01 2024

@author: Rishi
"""

import numpy as np
import pandas as pd

data = pd.read_csv("dataset.csv")

concepts = np.array(data.iloc[:,0:-1])
print(concepts)

target = np.array(data.iloc[:,-1])
print(target)

def learn(concepts, target):
    
    print('initialization of specific_h and general_h')
    specific_h = concepts[0].copy()
    print(specific_h)
   
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
   
    for i, h in enumerate(concepts):
        if target[i] == 'yes':
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
                print(specific_h)
            print(specific_h)
                   
        if target[i] == 'no':
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
      
                else:
                     general_h[x][x] = '?'
                     
    print('steps of candidate Elimination Algorithm', i+1)
    print(specific_h)
    print(general_h)      

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]

    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
 
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print('---------------final answer--------------\n')
print("final specific_hypothesis : ", s_final, sep="\n")  
print("final general_hypothesis : ", g_final, sep="\n")  