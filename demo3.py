import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
feature1 = [[1],[2],[3]]
values = [1,4,9]
plt.scatter(feature1,values, c='g')


regression1.fit(feature1,values)
print('Coefficient:{}'.format(regression1.coef_))
print('intrecept:{}'.format(regression1.intercept_))
range1 = [0,4]
plt.plot(range1, regression1.coef_*range1+regression1.intercept_, c='gray')
plt.show()
