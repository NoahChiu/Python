import tensorflow as tf
import numpy as np

a=np.array([5,4,3])
b = np.array([3,-1,2])
c = np.add(a,b)
print(c)

a2 = tf.constant([5,4,3])
b2 = tf.constant([3,-1,2])
c2 = tf.add(a,b)
print(c2)