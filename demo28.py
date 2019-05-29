import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

df1 = pd.read_csv('data\\sonar.all-data',header=None,prefix='X')
print(df1.shape)

#切割Data
data, labels = df1.iloc[:,:-1], df1.iloc[:,-1]
df1.rename(columns={'X60':'Label'}, inplace = True)

clas = KNeighborsClassifier(n_neighbors=3)
X_train, X_test,Y_train,Y_test = train_test_split(data, labels, test_size=0.2)
clas.fit(X_train, Y_train)
y_pred = clas.predict(X_test)

print('Score=',clas.score(X_test,Y_test))
resualt_cm = confusion_matrix(Y_test,y_pred)
print(resualt_cm)

scores = cross_val_score(clas, data, labels, cv=5, groups=labels)
print(scores)


