import numpy as np
from keras.datasets import boston_housing
from keras import models, layers
import matplotlib.pyplot as plt

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

print(train_data.shape, test_data.shape)

#nomalization
mean = train_data.mean(axis=0)#直的
std = train_data.std(axis=0)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model= build_model()
history = model.fit(train_data, train_target, validation_split=0.1, epochs=100, batch_size=10, verbose=1)

counter = 0
for i in range(0, len(test_data)):
    predict = model.predict(test_data[i].reshape(1,-1))
    print('Pridict price=%.3f, reference price=%.3f'%(predict, test_target[i]))
    counter += 1

plt.plot(test_target)
plt.show()


