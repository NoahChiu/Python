from sklearn import datasets, model_selection
from sklearn.decomposition import PCA

iris = datasets.load_iris()
data = PCA(n_components=2).fit_transform(iris.data)
print(data)

