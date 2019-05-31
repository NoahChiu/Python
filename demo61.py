import tensorflow as tf

a=tf.constant(100, name='a')
b=tf.constant(150, name='b')
mul = a*b

with tf.Session() as session:
    with tf.summary.FileWriter('c:\\logs', graph=session.graph) as writer:
        pass
    print(session.run(mul))