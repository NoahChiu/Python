import numpy as np

a = np.array([[1,2],[3,4]])
b = a
c = a.view()
d = a.copy()

print(a,'\n',b,'\n',c,'\n',d)
a[0][0] = 100
print(a,'\n',b,'\n',c,'\n',d)
a.shape = (4,)
print(a,'\n',b,'\n',c,'\n',d)