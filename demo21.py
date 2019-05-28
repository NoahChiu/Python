import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets

iris = datasets.load_iris()
df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
df1['species'] = np.array([iris.target_names[i] for i in iris.target])
print(df1)

seaborn.pairplot(df1,hue='species')
plt.show()