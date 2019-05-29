import numpy as np
from sklearn.naive_bayes import GaussianNB

X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])
clas = GaussianNB()
clas.fit(X,Y)
print(clas.predict([[-0.8,-0.8],[-2,2],[2,-2],[-2,-2]]))

clas2 = GaussianNB()
#後續可以再增加Fit條件, 但Y種類固定
clas2.partial_fit(X,Y,np.unique(Y))
print(clas2.predict([[-0.8,-0.8],[-2,2],[2,-2],[-2,-2]]))
clas2.partial_fit([[-0.7,-0.7]],[2])
print(clas2.predict([[-0.8,-0.8],[-2,2],[2,-2],[-2,-2]]))