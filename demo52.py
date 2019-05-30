import numpy as np
from keras.datasets import imdb
from keras import layers, models
from matplotlib import pyplot

#因為影評文字長短不一，因此可先給單字表，若有出現則為1 沒出現為0

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000) #取前10000筆

print(max([max(x) for x in train_data]))

word_index = imdb.get_word_index()

reverce_word_index = dict([(v,k) for k, v in word_index.items()])
decode = ''.join([reverce_word_index.get(i-3, '?') for i in train_data[0]])
print(decode)

def vectorize_seq(seq, dimension = 10000):
    resualts=np.zeros((len(seq), dimension))
    for i, seq in enumerate(seq):
        resualts[i, seq] = 1.
    return resualts

x_train = vectorize_seq(train_data)
x_test = vectorize_seq(test_data)
y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test,y_test))




