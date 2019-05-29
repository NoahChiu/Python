import tensorflow as tf
hello = tf.constant("hello")
session = tf.Session()
print(session.run(hello))
session.close()

