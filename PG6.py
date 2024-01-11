# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:41:44 2024

@author: Rishi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv("pima_indian.csv")
feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print('\n The total number of Training Data:',ytrain.shape)
print('\n The total number of Test Data:',ytest.shape)

clf = GaussianNB().fit(xtrain,ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData = clf.predict([[6,148,72,35,0,33.6,0.627,50]])

print('\n Confusion Matrix',metrics.confusion_matrix(ytest,predicted))
print('\n Accuracy',metrics.accuracy_score(ytest,predicted))
print('\n Precision',metrics.precision_score(ytest,predicted))
print('\n Recall',metrics.recall_score(ytest,predicted))
print('\n Predicted value for individual Test Data',predictTestData)