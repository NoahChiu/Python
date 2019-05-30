import numpy as np
from keras.layers import Dense
from keras.models import Sequential

dataset = np.loadtxt("data\\diabetes.csv", delimiter=",",skiprows=1)
print(type(dataset))
print(dataset.shape)

inputList = dataset[:,0:8] #inputList = dataset[:,0:-1]
outputList = dataset[:,8] #outputList = dataset[:,-1]
print("input shape:{}".format(inputList.shape))
print("output shape:{}".format(outputList.shape))

model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(inputList,outputList,epochs=200, batch_size=20)

scores = model.evaluate(inputList,outputList)
print('score:',scores)
print('matrics=',model.metrics_names)
print("%s:%.3f\n"%(model.metrics_names[0], scores[0]))
print("%s:%.3f\n"%(model.metrics_names[1], scores[1]))


