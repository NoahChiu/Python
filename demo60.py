import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer

csv = pd.read_csv('data\\bmi.csv')
csv['heigh'] = csv['heigh']/200
csv['weigh'] = csv['weigh']/100

encoder = LabelBinarizer()
transforLabel = encoder.fit_transform((csv['label']))
print(csv['label'][:10])
print(transforLabel[:10])

test_csv = csv[25000:]
test_pat = test_csv[['weigh','heigh']]
test_ans = transforLabel[25000:]

train_csv = csv[:25000]
train_pat = train_csv[['weigh','heigh']]
train_ans = transforLabel[:25000]

from keras.layers import Dense
model = keras.models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())

from keras import callbacks
borad = callbacks.TensorBoard(log_dir='c:\\logs',histogram_freq=1)
model.fit(train_pat, train_ans, batch_size=100, epochs=500, verbose=1, validation_data=(test_pat,test_ans),callbacks=[borad])