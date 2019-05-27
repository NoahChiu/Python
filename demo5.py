from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]],[1,4,8])

print("coefficient:{}, intercept:{}".format( reg.coef_, reg.intercept_))

resualt = reg.predict([[0.8,0.8],[2,1],[10,14]])
print("resualt:{}".format(resualt))

Score = reg.score([[0,0],[1,1],[2,2]],[1,4,8])
print("Score:{}".format(Score))