import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset = np.loadtxt("data\\diabetes.csv", delimiter=",",skiprows=1)
print(type(dataset))
print(dataset.shape)

inputList = dataset[:,0:8] #inputList = dataset[:,0:-1]
outputList = dataset[:,8] #outputList = dataset[:,-1]

def create_default_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim = 8, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

from keras.wrappers.scikit_learn import KerasClassifier

model1 = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers=['rmsprop','adam','sgd']
inits=['normal','uniform']
epochs = [50,100,150]
batches = [5,10,15]
param_grid = dict(optimizer = optimizers, epochs = epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model1, param_grid=param_grid)
grid_resualt = grid.fit(inputList, outputList)
print("best: %f, usin %s"%(grid_resualt.best_score_, grid_resualt.best_params_))

means = grid_resualt.cv_results_['mean_test_score']
stds = grid_resualt.cv_results_['std_test_score']
params = grid_resualt.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r"%(mean, stdev, param))
