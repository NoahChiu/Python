import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
np.random.seed(20190529)
X=np.r_[np.random.randn(50,2)+[2,2], np.random.randn(50,2)+[0,-2],np.random.randn(50,2)+[-2,2]]

k=1
kmean = KMeans(n_clusters=k)
kmean.fit(X)
print("kmean center:",kmean.cluster_centers_)
print("label:",kmean.labels_)
print("iteration:",kmean.n_iter_) #步數
print("cost:",kmean.inertia_) #Cost