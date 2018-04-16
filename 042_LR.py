'''
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, 
fit_intercept=True,intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’,verbose=0, warm_start=False, n_jobs=1)

#参数详细说明：https://blog.csdn.net/jark_/article/details/78342644

对于多类：

（1）如果参数“multi_class=‘ovr’ ”，分类方法使用one-vs-rest (OvR，一对多)；

（2）如果参数“multi_class=‘multinomial’ ”，损失函数使用cross-entropy loss（目前支持‘multinomial’的优化方法有‘lbfgs’，‘sag’，‘newton-cg’）
'''

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
lr.predict_proba(X_test_std[0,:]) # 查看第一个测试样本属于各个类别的概率
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
