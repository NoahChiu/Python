import numpy as np

a = np.zeros((10,2))
print(a)
b = a.T
print(b)
c = b.view()
print(c)
d = np.reshape(b,(5,4))
print(d)
e = np.reshape(b,(20,))
print(e)
f = np.reshape(b,(20,-1))
print(f)
g = np.reshape(b,(20,1))
print(g)
h = np.reshape(b,(-1,20))
print(h)
print(d.shape, e.shape, f.shape, g.shape, h.shape)