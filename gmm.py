# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:42:08 2020

@author: Smarak Pradhan
"""

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Clustering_gmm.csv')

 
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)


labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()
