from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

print(list(iris.keys()))

# V.S. petal width
X = iris.data[:,3:] #Petal Width
Y = (iris.target == 2).astype(np.int) #Type = 2
reg = LogisticRegression()
reg.fit(X,Y)
print(reg.intercept_,reg.coef_)

X_seq = np.linspace(0,2.5,1000).reshape(-1,1)
Pre_Y = reg.predict_proba(X_seq)
print(Pre_Y)

plt.plot(X,Y, 'gs')
plt.plot(X_seq, Pre_Y[:,1],"b--",label="iris-virginice")#預測是Type2機率
plt.plot(X_seq, Pre_Y[:,0],"r--",label="Non iris-virginice")#預測不是Type2機率
plt.xlabel("Petal Width", fontsize = 14)
plt.ylabel("Probability", fontsize=14)
plt.legend()
plt.show()
