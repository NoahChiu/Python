import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])
#Y = np.array([1,3,2,1,3,2])
x_min,x_max = -4,4
y_min, y_max = -4,4
h=0.025
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
clf1 = GaussianNB()
clf1.fit(X,Y)
z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)

plt.pcolormesh(xx,yy,z)

xb, yb, xr, yr = [],[],[],[]

index = 0
for i in range(0,len(Y)):
    if Y[i] == 1:
        print("B eqaul to", X[i,:])
        xb.append(X[i,0])
        yb.append(X[i,1])
    elif Y[i] == 2:
        print("R eqaul to", X[i,:])
        xr.append(X[i,0])
        yr.append(X[i,1])
print(X)
print(X[:,0])
plt.scatter(xb,yb,color='b',label='Blue')
plt.scatter(xr,yr,color='r',label='Red')
plt.legend()
plt.show()


