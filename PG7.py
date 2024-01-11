# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:43:28 2024

@author: Rishi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn  import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']

dataset = pd.read_csv("Kdataset (1).csv",names=names)

x = dataset.iloc[:,:-1]

label = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

y = [label[c] for c in dataset.iloc[:,-1]]

plt.figure(figsize=(14,7))
colormap = np.array(['red','lime','black'])

#REALPLOT
plt.subplot(1,3,1)
plt.title('Real_plot')
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y])

#K-PLOT
model = KMeans(n_clusters=3,random_state=0).fit(x)
plt.subplot(1,3,2)
plt.title('KMeans')
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[model.labels_])
print('The accuracy score of KMean:',metrics.accuracy_score(y,model.labels_))
print('The Confusion matrix of KMean:\n',metrics.confusion_matrix(y,model.labels_))

#GMM PLOT
gmm = GaussianMixture(n_components=3,random_state=0).fit(x)
y_cluster_gmm = gmm.predict(x)
plt.subplot(1,3,3)
plt.title('GMM Classification')
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y_cluster_gmm])
print('The accuracy score of EM:',metrics.accuracy_score(y,y_cluster_gmm))
print('The Confusion matrix of EM:\n',metrics.confusion_matrix(y,y_cluster_gmm))

plt.show()