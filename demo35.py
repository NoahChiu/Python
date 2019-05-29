from numpy import array, cov, mean
from numpy.linalg import eig

A= array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
M = mean(A.T, axis=1)

C=A-M
V = cov(C.T)
values,vectors = eig(V)
print('vectors=', vectors)
print('values=',values)
P = vectors.T.dot(C.T)
print(P)