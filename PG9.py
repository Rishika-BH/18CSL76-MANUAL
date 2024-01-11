# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:01:44 2024

@author: Rishi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point,xmat,k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - x[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights


def localWeight(point,xmat,ymat,k):
    wei = kernel(point, xmat, k)
    W = (x.T * (wei*x)).I * (x.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i], xmat, ymat, k)
    return ypred

data = pd.read_csv("tips.csv")
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
x = np.hstack((one.T, mbill.T))

ypred = localWeightRegression(x, mtip, 0.5)
SortIndex = x[:,1].argsort(0)
xsort = x[SortIndex][:,0]

fig = plt.figure();
ax = fig.add_subplot(1,1,1)

ax.scatter(bill,tip,color = 'green')
ax.plot(xsort[:,1], ypred[SortIndex],color = 'red', linewidth = 5)

plt.xlabel('Total bILL')
plt.ylabel('Tip')
plt.show()
