import matplotlib.pyplot as plt
import tensorflow as tf
import keras.datasets as datasets

(train_image, train_label), (test_image, test_label) = \
    datasets.mnist.load_data()

def plotimage(index):
    plt.title("train image marked as %d"%train_label[index])
    plt.imshow(train_image[index], cmap='binary')
    plt.show()
    pass

def plottesimage(index):
    plt.title("test image marked as %d"%test_label[index])
    plt.imshow(test_image[index], cmap='binary')
    plt.show()
    pass

plotimage(500)
plottesimage(500)