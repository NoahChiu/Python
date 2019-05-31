import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models, layers
import numpy as np

(train_data, train_label), (test_data, test_label) = reuters.load_data(num_words=10000)

print(np.unique(train_label))

word_index = reuters.get_word_index()
reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])
decode = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

def vectorize_sequence(sequences, dimension=10000):
    resualt = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        resualt[i, sequence] = 1.
    return resualt

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)


def to_one_hot(labels, dimension=46):
    resualt = np.zeros((len(labels), dimension))
    for i, labels in enumerate(labels):
        resualt[i, labels] = 1.
    return resualt
one_hot_train_label = to_one_hot(train_label)
one_hot_test_label = to_one_hot(test_label)

print(one_hot_test_label[:10])
print(one_hot_train_label[:10])

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, one_hot_train_label, epochs=20, batch_size=20,validation_data=(x_test,one_hot_test_label))

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss,label = 'Training Loss')
plt.plot(val_loss, color='r', label='Validate Loss')
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(acc,label = 'Training Accuracy')
plt.plot(val_acc, color='r', label='Validate Accuracy')
plt.show()
