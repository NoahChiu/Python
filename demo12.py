from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(type(iris))

X = iris.data
Y = iris.target
labels = ["SL","SW","PL","PW"]
print(type(X),type(Y))
print(X.shape)

#Plot the picture
counter =1
for i in range(0,4):
    for j in range(i+1,4):
        plt.figure(counter, figsize = (8,6))
        counter += 1
        xData = X[:,i]
        yData = X[:,j]
        x_min, x_max = xData.min()-0.5, xData.max()+0.5
        y_min, y_max = yData.min() - 0.5, yData.max() + 0.5

        plt.clf()
        plt.scatter(xData, yData, c=Y, cmap = plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()