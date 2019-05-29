import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X=np.r_[np.random.randn(50,2)+[2,2], np.random.randn(50,2)+[0,-2],np.random.randn(50,2)+[-2,2]]

kmean = KMeans(n_clusters= 4)
kmean.fit(X)

colors = ['c','m','y','k']
markers = ['o','v','*','x']

for i in range(4):
    dataX = X[kmean.labels_ == i]
    plt.scatter(dataX[:,0],dataX[:,1],c=colors[i],marker=markers[i])
    print(dataX.shape)
plt.show()


