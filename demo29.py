import pandas as pd
from sklearn import datasets, model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
data = iris.data
target = iris.target

clas = KNeighborsClassifier(n_neighbors=3)
score = model_selection.cross_val_score(clas,data,target,cv=5)
print(score)

