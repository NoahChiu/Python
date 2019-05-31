import tensorflow as tf

a=tf.constant(3, name='a')
b=tf.constant(5,name='b')
c=tf.constant(7,name='c')
x=tf.Variable(0,name='var')

cal = a+b*(b+c)**a
assign_op = tf.assign(x,cal)
with tf.Session() as session:
    with tf.summary.FileWriter('c:\\logs', graph=session.graph) :
        session.run(assign_op)
        print(session.run(x))