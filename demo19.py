import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings("ignore")


X=[[0,0],[1,1],[0,1],[1,0]]
Y = [0,0,1,1]
color = ['red','green']
marker = ['o','d']

index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0],X[index][1],c=color[type],marker = marker[type])
    index +=1
plt.show()

clas = tree.DecisionTreeClassifier()
clas.fit(X,Y)
export_graphviz(clas, out_file='graph\\demo19.dot',filled=True,rounded=True,special_characters=True)