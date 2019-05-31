from keras import backend
from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#設定影像處理順序
backend.set_image_dim_ordering('th') #thano

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test = x_test.reshape(x_test.shape[0],1,28,28).astype('float32')

x_train = x_train/255 #normalization
x_test = x_test/255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(1, 28, 28), activation='relu')) #3*3 filter 32 layers
    model.add(MaxPooling2D(2,2)) #尺寸縮小一半
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten()) #攤平
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_class,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

model = baseline_model()
model.fit(x_train, y_train, epochs=10, batch_size=100)




