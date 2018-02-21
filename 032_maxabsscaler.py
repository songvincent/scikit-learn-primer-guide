#它通过除以最大值将训练集缩放至[-1,1]。这意味着数据已经以０为中心或者是含有非常非常多０的稀疏数据。
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],  
                     [ 2.,  0.,  0.],  
                    [ 0.,  1., -1.]])  
max_abs_scaler = preprocessing.MaxAbsScaler()  
X_train_maxabs = max_abs_scaler.fit_transform(X_train)  
# doctest +NORMALIZE_WHITESPACE^, out: array([[ 0.5, -1.,  1. ], [ 1. , 0. ,  0. ],       [ 0. ,  1. , -0.5]])  
X_test = np.array([[ -3., -1.,  4.]])  
X_test_maxabs = max_abs_scaler.transform(X_test) #out: array([[-1.5, -1. ,  2. ]])  
print(max_abs_scaler.scale_ ) #out: array([ 2.,  1.,  2.])  
print(X_test_maxabs)