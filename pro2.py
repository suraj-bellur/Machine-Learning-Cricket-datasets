import pandas as pd
import numpy as np

dataset=pd.read_csv('t20.csv')

to_drop=['Player','Mat','Inns','Runs','HS','Ave','BF']
dataset.drop(to_drop,inplace=True,axis=1)
#print(dataset)

X=dataset.iloc[:,[1,2,3,4,5]].values
#print(X)
y=dataset.iloc[:,0].values
#print(y)


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#print(y_pred)

y_pred1=regressor.predict([[2,10,3,147,56]])
print(y_pred1)
