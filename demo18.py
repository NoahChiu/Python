from sklearn import tree

X=[[0,0],[1,1]]
Y = [0,1]
clas = tree.DecisionTreeClassifier()
clas.fit(X,Y)

print(clas.predict([[2,2],[2,-2],[-2,2],[-2,-2]]))