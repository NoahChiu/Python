import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(diabetes.data.shape)
print(diabetes.target.shape)

data_train = diabetes.data[:-50]
target_train = diabetes.target[:-50]

data_test = diabetes.data[-50:]
target_test = diabetes.target[-50:]

reg = linear_model.LinearRegression()
reg.fit(data_train, target_train)

print("regression finish")
print("Score", reg.score(data_test, target_test))

for i in range(-50, 0):
    dataarray = data_test[i].reshape(1,-1)
    print("predict/actual",reg.predict(dataarray)[0], target_test[i])
    #print(data_test[i].reshape(1,-1))