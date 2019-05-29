import tensorflow as tf

def computeArea(sides):
    a = sides[:,0]
    b = sides[:,1]
    c = sides[:,2]
    s = (a + b +c)/2
    areaSquare = s*(s-a)*(s-b)*(s-c)
    return areaSquare**0.5

with tf.Session() as session:
    sides = tf.placeholder(tf.float32, shape=(None,3))
    area = computeArea(sides)
    resualt = session.run(area, feed_dict={
        sides:[[3.0,4.0,5.0],[6.0,6.0,6.0],[7.0,8.0,9.0]]
    })
    print(resualt)