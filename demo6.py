import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#產生 10 筆資料, 且有5個欄位
RegData1 = datasets.make_regression(10, 5)
print(RegData1[0], RegData1[1])


#產生 200 個radom number, noise = 105
RegData2 = datasets.make_regression(200,1, noise = 10)
print(RegData2[0], RegData1[1])

reg = linear_model.LinearRegression()
reg.fit(RegData2[0], RegData2[1])
print("coef:{}, intercept:{}".format(reg.coef_, reg.intercept_))
print("Score:{}".format(reg.score(RegData2[0], RegData2[1])))

range = [-3, 3]
plt.plot(range, reg.coef_*range+reg.intercept_)
plt.scatter(RegData2[0], RegData2[1], c='r')
plt.show()