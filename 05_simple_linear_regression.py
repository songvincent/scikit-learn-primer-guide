'''
回归是根据指定的对象的连续值属性与此连续值，如股票价格的预测。
Scikit-learn 实现的回归算法包括支持向量回归（SVR）、脊回归（Ridge Regression）、套索lasso 回归、弹性网络（Elastic Net）和 Bayes 回归等等。

'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0 )

#很简单的线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#执行到这里就结束了
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()