import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

dataset=pd.read_csv('cricket.csv')

to_drop=['batsman']
dataset.drop(to_drop,inplace=True,axis=1)

X=dataset.iloc[:,:-1].values
#print(X)
y=dataset.iloc[:,5].values
#print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
#print(y_pred)

y_pred1=logreg.predict([[100,2000,2500,45,147]])
#print(y_pred1)

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
#print(cnf_matrix)

sn.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title('confusion matrix')
plt.xlabel('actual label')
plt.ylabel('predicted label')
plt.show()
#print(metrics.accuracy_score(y_test,y_pred))
#print(metrics.precision_score(y_test,y_pred))
#print(metrics.recall_score(y_test,y_pred))




