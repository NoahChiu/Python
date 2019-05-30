import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = np.loadtxt("data\\diabetes.csv", delimiter=",",skiprows=1)
print(type(dataset))
print(dataset.shape)

inputList = dataset[:,0:8] #inputList = dataset[:,0:-1]
outputList = dataset[:,8] #outputList = dataset[:,-1]

feature_train, feature_test, label_train, label_test = train_test_split(inputList, outputList, test_size=0.1)
print("input shape:{}".format(inputList.shape))
print("output shape:{}".format(outputList.shape))

model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(feature_train,label_train,epochs=2000, batch_size=20, validation_data=(feature_test, label_test))

loss = history.history.get('loss')
acc = history.history.get('acc')
print(loss)
print(acc)

plt.plot(loss,label='Loss')
plt.plot(acc,color='r',label='Acc')
plt.legend()
plt.show()

