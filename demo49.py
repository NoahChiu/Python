from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import KFold, cross_val_score


dataFrame = read_csv('data\\iris.data',header=None)
dataset = dataFrame.values #value =>表格轉陣列

feature = dataset[:,0:4].astype(float)
label = dataset[:,-1]

encoder = LabelEncoder()
encoder.fit(label)
encoded_y = encoder.transform(label)#將結果轉乘0,1,...
print(encoded_y[:10], encoded_y[50:60], encoded_y[100:110])
dummy_y = np_utils.to_categorical(encoded_y)#將結果轉乘[1,0,0],[0,1,0],[0,0,1]
print(dummy_y[:3], dummy_y[50:53], dummy_y[100:103])

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10,verbose=0)
kfold = KFold(n_splits=3, shuffle=True)
resualt = cross_val_score(estimator, feature, dummy_y, cv=kfold)
print("acc:%.4f, std:%.4f"%(resualt.mean(), resualt.std()))