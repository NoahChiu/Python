import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

dataset = np.loadtxt("data\\diabetes.csv", delimiter=",",skiprows=1)
print(type(dataset))
print(dataset.shape)

inputList = dataset[:,0:8] #inputList = dataset[:,0:-1]
outputList = dataset[:,8] #outputList = dataset[:,-1]
#generate k-fold
fivefold = StratifiedKFold(n_splits=5, shuffle=True)
totalscroes = []


model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

for train, test in fivefold.split(inputList,outputList):
    model.fit(inputList[train],outputList[train],epochs=200, batch_size=20)
    scores = model.evaluate(inputList[test], outputList[test])
    totalscroes.append(scores[1]*100)
print("total 5 resualt mean:%.3f, std:%.3f\n"%(np.mean(totalscroes), np.std(totalscroes)))


