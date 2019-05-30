import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score

dataset = np.loadtxt("data\\diabetes.csv", delimiter=",",skiprows=1)
print(type(dataset))
print(dataset.shape)

inputList = dataset[:,0:8] #inputList = dataset[:,0:-1]
outputList = dataset[:,8] #outputList = dataset[:,-1]

def create_default_model():
    model = Sequential()
    model.add(Dense(64, input_dim = 8, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

from keras.wrappers.scikit_learn import KerasClassifier

model1 = KerasClassifier(build_fn=create_default_model, epochs=200, batch_size=20)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
resualt = cross_val_score(model1, inputList, outputList, cv=fiveFold)
print("mean=%.3f, std=%.3f"%(resualt.mean(), resualt.std()))


