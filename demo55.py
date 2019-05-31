import keras.utils as utils

orig = 3
NUM_DIGIT = 20
print(orig)

convert = utils.to_categorical((orig, NUM_DIGIT))
print(convert)

type = 10
conver2 = utils.to_categorical(orig, type)
print(conver2)

orig2 = 0
convert3 = utils.to_categorical(orig2, type)
print(convert3)