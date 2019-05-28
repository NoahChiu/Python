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

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df1[iris.feature_names],iris.target,test_size=0.5,stratify=iris.target)

from sklearn.ensemble import RandomForestClassifier
ref = RandomForestClassifier(n_estimators=100, oob_score=True)#100 Tree and Show Scores
ref.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
pre = ref.predict(X_test)
acc = accuracy_score(Y_test, pre)

print('00B Score:{:.3}'.format(ref.oob_score_))
print('Accuracy:{:.3}'.format(acc))

from sklearn.metrics import confusion_matrix
cm = pd.SparseDataFrame(confusion_matrix(Y_test,pre),columns=iris.target_names,index=iris.target_names)
seaborn.heatmap(cm,annot=True)
plt.show()


