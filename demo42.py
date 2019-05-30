import tensorflow as tf
v1 = [-2.0,-1.0,-1.005,0.5,2,5]
res = tf.nn.relu(v1)
res2 = tf.nn.sigmoid(v1)

with tf.Session() as session:
    print('Relu resualt:',session.run(res)) #篩掉負數
    print('Sigmoid:',session.run(res2))