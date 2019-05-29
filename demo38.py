import tensorflow as tf

a=tf.placeholder(dtype=tf.int32, shape=(None,))#placeholde => 給定變數
b=tf.placeholder(dtype=tf.int32, shape=(None,))
c=tf.add(a,b)


with tf.Session() as session:
    resualt = session.run(c, feed_dict={
    a:[3,4,5],
    b:[5,6,7]})
    print(resualt)

