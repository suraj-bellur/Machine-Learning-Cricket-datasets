import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("t20.csv")

to_drop=['Player','Inns','Runs','HS','Ave','BF','SR','100s','50s','0s','4s']
dataset.drop(to_drop,inplace=True,axis=1)
#print(dataset)

X=dataset.iloc[:,:-1].values
#print(X)
y=dataset.iloc[:,1].values
#print(y)


from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25,
                                                    random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)
#print(y_pred)

y_pred1=regressor.predict([[151]])
#print(y_pred1)

check=pd.DataFrame(y_pred,y_test)
#print(check)

train=plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('number of sixes (training set)')
plt.xlabel('matches')
plt.ylabel('sixes')
plt.show()

test=plt
test.scatter(X_test,y_test,color='green')
test.plot(X_test,regressor.predict(X_test),color='blue')
test.title('number of sixes (test set)')
test.xlabel('matches')
test.ylabel('sixes')
test.show()

