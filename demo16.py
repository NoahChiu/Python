from sklearn import datasets, model_selection, svm
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
svc = svm.SVC()
scores = model_selection.cross_val_score(svc,iris.data,iris.target,cv=5)
print(scores)
print("accuracy", scores.mean())