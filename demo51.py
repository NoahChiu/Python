import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot

(x_train, y_train), (x_test, y_test) = imdb.load_data()

x=np.concatenate((x_train, x_test), axis=0)
y=np.concatenate((y_train,y_test), axis=0)

print(len(np.unique(np.hstack(x)))) #將x中影評結合成一列

resualt = [len(x) for x in x]
print("mean length: %.3f, std:%.3f"%(np.mean(resualt), np.std(resualt)))

pyplot.subplot(1,2,1)
pyplot.boxplot(resualt)
pyplot.subplot(1,2,2)
pyplot.hist(resualt)
pyplot.show()


