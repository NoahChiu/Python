import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, model_selection
from sklearn.decomposition import PCA

iris = datasets.load_iris()
species = iris.target
data = iris.data

fig = plt.figure(1, figsize=(8,8))
ax = Axes3D(fig, elev=-150, azim = 110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(X_reduced[:,0],X_reduced[:,1],X_reduced[:,2], c=species, cmap=plt.cm.Paired)
ax.set_title("first three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
