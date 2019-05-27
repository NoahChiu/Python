import matplotlib.pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
feature1 = [[0,1],[1,3],[2,8]]
values = [1,4,5.5]

regression1.fit(feature1,values)
print('Coefficient:{}'.format(regression1.coef_))
print('intrecept:{}'.format(regression1.intercept_))
print('First elementory={}, Second elementory={}'.format(regression1.coef_[0], regression1.coef_[1]))
