# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:45:22 2024

@author: Rishi
"""


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



names = ['sepal_length','sepal_width','petal_length','petal_width','class']
dataset = pd.read_csv("Kdataset (1).csv")

X = dataset.iloc[:,:-1]
print('sepal_length','sepal_width','petal_length','petal_width')
print(X.head())

y = dataset.iloc[:,-1]
print('Target Value')
print(y.head())

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5).fit(Xtrain,ytrain)
ypred = classifier.predict(Xtest)
i = 0

print("\n---------------------------------------------------")
print("%-25s %-25s %-25s" % ('Original Label','Predicted Label','Correct/Wrong'))
print("\n---------------------------------------------------")

for label in ytest:
    print('%-25s %-25s' % (label,ypred[i]),end=" ")
    if label == ypred[i]:
        print('%-25s' % 'Correct')
    else:
        print('%-25s' %'Wrong')
    i = i+1
    
print("\n---------------------------------------------------")
print("Confusion Matrix:\n",metrics.confusion_matrix(ytest,ypred))
print("\n---------------------------------------------------")
print("Classification Report:\n",metrics.classification_report(ytest,ypred))
print("\n---------------------------------------------------")
print("Accuracy of the classifier is", metrics.accuracy_score(ytest,ypred))
print("\n---------------------------------------------------") 