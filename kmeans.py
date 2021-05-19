# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:33:04 2020

@author: Smarak Pradhan
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Clustering_gmm.csv')

plt.figure(figsize=(7,7))
plt.scatter(data["Weight"],data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)


pred = kmeans.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = pred
frame.columns = ['Weight', 'Height', 'cluster']


color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()


