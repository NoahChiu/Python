import tensorflow as tf
import keras
import numpy as np

(train_image, train_label), (test_image, test_label) = keras.datasets.mnist.load_data()

FLATTEN_DIM=28*28
TRAINING_SIZE=len(train_image)
TEST_SIZE=len(test_image)

trainImage = np.reshape(train_image,(TRAINING_SIZE, FLATTEN_DIM))
testImage = np.reshape(test_image,(TEST_SIZE, FLATTEN_DIM))

trainImage = trainImage.astype((np.float32))
testImage = testImage.astype((np.float32))

trainImage /= 255
testImage /= 255

NUM_DIGITS = 10

trainlabel = keras.utils.to_categorical(train_label, NUM_DIGITS)
testlabel = keras.utils.to_categorical(test_label, NUM_DIGITS)

model = keras.Sequential()
model.add(keras.layers.Dense(units=256, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
print(model.summary())
#make tensorboard callback
tb = keras.callbacks.TensorBoard(log_dir='c:\\logs',histogram_freq=0, write_graph=True, write_images=True)
model.fit(trainImage, trainlabel, epochs=10)

loss, accuracy = model.evaluate(testImage, testlabel)
print('accuracy={}'.format(accuracy))

p1 = model.predict(testImage)
p2 = model.predict_classes(testImage)
p3 = model.predict_proba(testImage)


