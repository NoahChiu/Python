import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
enighbor = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = enighbor.kneighbors(X, return_distance= True)

#與鄰居彼此間距離 distance[self, nearest neighbor]
print(distances)
#最近鄰居的index indices from distance
print(indices)
#彼此最近顯示1 其餘顯示0
print(enighbor.kneighbors_graph(X).toarray())

