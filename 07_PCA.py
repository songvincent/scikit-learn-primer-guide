import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


#划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#归一化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#数据降维
#对降维的解释，这里还是蛮全面的：http://blog.jobbole.com/109015/
#先创建一个PCA对象，其中参数n_components表示保留的特征数，默认为1。如果设置成‘mle’,那么会自动确定保留的特征数
from sklearn.decomposition import PCA
pca = PCA(n_components = 2 )
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# print(X_train)
# print(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)