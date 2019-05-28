import numpy as np

a = np.array([[1,2],[3,4]])
print(a)
print(a.shape)

a.shape = (4,-1)
print(a)

b=a.view()
print(b)
b.shape = (1,4)
print(b)

c = a
print(c)
c.shape = (2,2)
print(c)

d = a.copy()
print(d)
d.shape = (4,)
print(d)