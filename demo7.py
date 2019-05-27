import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

RegData = datasets.make_regression(10, 6, noise=10)
x = RegData[0]
y = RegData[1]

for i in range(0,6):
    plt.scatter(x[:,i],y)
    #plt.show()

r1 = sorted(x, key=lambda tup:tup[0])
r2 = sorted(x, key=lambda tup:tup[1])
r3 = sorted(x, key=lambda tup:tup[2])
r4 = sorted(x, key=lambda tup:tup[3])
r5 = sorted(x, key=lambda tup:tup[4])
r6 = sorted(x, key=lambda tup:tup[5])
