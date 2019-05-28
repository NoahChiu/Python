from sklearn import datasets, model_selection, svm, tree
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
data = iris.data
target = iris.target

tree = tree.DecisionTreeClassifier()
score = model_selection.cross_val_score(tree,data,target,cv=5)
print(score)
