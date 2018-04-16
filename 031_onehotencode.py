'''
独热编码即 One-Hot 编码，又称一位有效编码，
其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，就变成了m个二元特征。
并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。

1.使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。 
2.将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，
而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。 
3.将离散型特征使用one-hot编码，可以会让特征之间的距离计算更加合理。
'''

import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder() #实例化
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]]) #fit

print ("enc.n_values_ is:",enc.n_values_)
#enc.n_values_ is: [2 3 4]

print ("enc.feature_indices_ is:",enc.feature_indices_)
#enc.feature_indices_ is: [0 2 5 9] 2 2+3 2+3+4

print (enc.transform([[0, 1, 1]]).toarray())
print (enc.transform([[0, 1, 1]]))
#[[ 1.  0.  0.  1.  0.  0.  1.  0.  0.]]
以上的列数 9=2+3+4 然后取1的位置
就是表示的值

'''
(0, 6)        1.0
(0, 3)        1.0
(0, 0)        1.0
稀疏矩阵形式，
'''
'''
基于树的方法不需要进行特征的归一化。
例如随机森林，bagging与boosting等方法。
如果是基于参数的模型或者基于距离的模型，因为需要对参数或者距离进行计算，都需要进行归一化。
'''

# print(enc.transform([[0,2,3]]))
