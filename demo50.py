

scores=[4.0,1.0,2.1,1.0,1.0]
print(scores)
import numpy as np

def SoftMax(x):
    y= np.array(x)
    return np.exp(x) / np.sum(np.exp(y), axis=0)

print(SoftMax(scores))


import tensorflow as tf

resualt = tf.nn.softmax(scores)
with tf.Session() as session:
    print(session.run(resualt))