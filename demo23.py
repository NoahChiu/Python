import numpy as np
import matplotlib.pyplot as plt
X=np.r_[np.random.randn(50,2)+[2,2], np.random.randn(50,2)+[0,-2],np.random.randn(50,2)+[-2,2]]
#創造Data
print(X)
[plt.scatter(e[0],e[1],c='g',s=7) for e in X]
#分成三群
k=3
#任意選中心點
C_x = np.random.randint(np.min(X),np.max(X),size=k)
C_y = np.random.randint(np.min(X),np.max(X),size=k)
C = np.array(list(zip(C_x,C_y)),dtype = np.float32)
plt.scatter(C_x,C_y,marker='*',c='#C02244')

plt.show()

#計算距離
def dist(a,b,axis=1):
    return np.linalg.norm(a-b, axis=axis)

#紀錄上次距離
#第一次給0

