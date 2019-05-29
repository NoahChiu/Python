import numpy as np
import matplotlib.pyplot as plt
X=np.r_[np.random.randn(50,2)+[2,2], np.random.randn(50,2)+[0,-2],np.random.randn(50,2)+[-2,2]]
#創造Data
print(X)
[plt.scatter(e[0],e[1],c='g',s=7) for e in X]
#分成三群
k=3
#任意選中心點
#C_x = np.random.randint(np.min(X),np.max(X),size=k)
#C_y = np.random.randint(np.min(X),np.max(X),size=k)
#C_x = np.random.uniform(np.min(X[:,0]),np.max(X[:,1]),size=k)
#C_y = np.random.uniform(np.min(X[:,0]),np.max(X[:,1]),size=k)

#直接從現有點找三點
C_x = np.zeros(k)
C_y = np.zeros(k)
counter=0
for i in np.random.choice(range(0,150),size = 3, replace=False): #不取重複值 replace=False
    C_x[counter] = X[i,0]
    C_y[counter] = X[i,1]
    counter += 1

C = np.array(list(zip(C_x,C_y)),dtype = np.float32)
plt.scatter(C_x,C_y,marker='*',c='#C02244')

plt.show()

#計算距離
def dist(a,b,axis=1):
    return np.linalg.norm(a-b, axis=axis)

def plot_kmean(current_cluster, delta):
    colors = ['r','g','b','c','m']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if current_cluster[j] == i])
        ax.scatter(points[:,0],points[:,1],c=colors[i])
    ax.scatter(C[:, 0], C[:, 1],marker='*', c='#C02244')
    plt.title('delta will be:%.4f' %delta)
    plt.plot()
    plt.show()

#紀錄上次距離
#第一次距離給0
from copy import deepcopy
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C,C_old,None)

while delta != 0:
    print('start a new iteration')
    for i in range(len(X)):
        distances = dist(X[i],C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C,C_old,None)
    plot_kmean(clusters, delta)

