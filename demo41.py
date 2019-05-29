
import tensorflow as tf

def computeArea(sides):
    a = sides[:,0]
    b = sides[:,1]
    c = sides[:,2]
    s = (a + b +c)/2
    areaSquare = s*(s-a)*(s-b)*(s-c)
    return areaSquare**0.5

tf.enable_eager_execution() #這行可以讓tensorflow執行程式 但要放在執行前
area = computeArea(tf.constant([[3.0,4.0,5.0],[6.0,6.0,6.0],[7.0,8.0,9.0]]))
print(area)