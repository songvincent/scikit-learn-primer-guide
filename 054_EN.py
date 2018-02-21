'''
ElasticNet 是一种使用L1和L2先验作为正则化矩阵的线性回归模型.这种组合用于只有很少的权重非零的稀疏模型，
比如:class:Lasso, 但是又能保持:class:Ridge 的正则化属性.我们可以使用 l1_ratio 参数来调节L1和L2的凸组合(一类特殊的线性组合)。

当多个特征和另一个特征相关的时候弹性网络非常有用。Lasso 倾向于随机选择其中一个，而弹性网络更倾向于选择两个.
在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许在循环过程（Under rotate）中继承 Ridge 的稳定性.
'''

from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston

boston=load_boston()

lr = ElasticNet()
lr.fit(boston.data,boston.target)

y_pre=lr.predict([[ 2.73100000e-02 , 0.00000000e+00 , 7.85000000e+00 , 0.00000000e+00,
   4.85000000e-01 ,  6.48100000e+00  , 7.89300000e+01  , 4.910000e+00,
   2.00000000e+00  , 3.78000000e+02 ,  1.78000000e+01 ,  3.96900000e+02,
   9.14000000e+00]])
print(y_pre)
