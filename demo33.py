from sklearn import datasets, model_selection
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = PCA(n_components=2).fit_transform(iris.data)
print(data)

datamax = data.max(axis=0)+1
datamin = data.min(axis=0)-1
print(datamax,datamin)

n = 4000
X, Y = np.meshgrid(np.linspace(datamin[0],datamax[0],n), np.linspace(datamin[1],datamax[1],n))

svc = svm.SVC(C=2000)
svc.fit(data, iris.target)
z = svc.predict(np.c_[X.ravel(),Y.ravel()])

plt.contour(X,Y,z.reshape(X.shape),levels=[-0.5,0.5,1.5,2.5],colors=['r','g','b','y'])

for i, c in zip([0,1,2],['r','g','b']):
    d = data[iris.target == i]
    plt.scatter(d[:,0],d[:,1],c=c)
plt.show()
