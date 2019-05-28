import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([1,1,1,2,2,2])
clas = SVC()
clas.fit(x,y)
print("predict=", clas.predict([[-0.8,-1],[4,4],[3,-3],[-3,3]]))
