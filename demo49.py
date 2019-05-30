from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

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
