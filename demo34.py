from numpy import array
from sklearn.decomposition import PCA

A= array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
pca = PCA(2).fit(A)
print('component=',pca.components_)
print('variance=', pca.explained_variance_)

B = pca.transform(A)
print(B)