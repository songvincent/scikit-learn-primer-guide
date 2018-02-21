import numpy as np
from sklearn.cross_validation import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(4, n_folds=2)

'''
n：int 数据对象的个数；
n_folds：int，默认为3，折数；
shuffle：boolean，可选，决定是否重新排列数据。
'''

print(len(kf))
print(kf)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
	
#KFold 的使用方法是直接用 for 循环来迭代 KFold 对象，获取其下标。