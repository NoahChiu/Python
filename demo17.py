from sklearn import datasets, model_selection, svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()

pca = PCA(n_components= 2)#降成2維
data = pca.fit(iris.data).transform(iris.data)

print(iris.data)
print(data)

datamax = data.max(axis=0)+1 #axis相當於比較直行的值
datamin = data.min(axis=0)-1

n=2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),np.linspace(datamin[1],datamax[1],n))
#meshgrid:將數值由大到小填滿

#kernel = 'linear','rbf','poly'
#c = 100
svc = svm.SVC(kernel='rbf', C=100)
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()]) #np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
                                             #ravel 多維轉一維
plt.contour(X,Y,Z.reshape(X.shape), color = 'K')
for c,s in zip([0,1,2],['.','+','*']):
    d = data[iris.target == c]
    plt.scatter(d[:,0], d[:,1],c='k',marker=s)
plt.show()


