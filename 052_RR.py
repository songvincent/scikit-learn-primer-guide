
'''
当设计矩阵X存在多重共线性的时候（数学上称为病态矩阵），最小二乘法求得的参数w在数值上会非常的大，而一般的线性回归其模型是 y=wTx ，
显然，就是因为w在数值上非常的大，所以，如果输入变量x有一个微小的变动，其反应在输出结果上也会变得非常大，这就是对输入变量总的噪声非常敏感的原因。
如果能限制参数w的增长，使w不会变得特别大，那么模型对输入w中噪声的敏感度就会降低。这就是脊回归和套索回归（Ridge Regression and Lasso Regrission）的基本思想。

为了限制模型参数w的数值大小，就在模型原来的目标函数上加上一个惩罚项，这个过程叫做正则化（Regularization）。

如果惩罚项是参数的l2范数，就是脊回归(Ridge Regression)

如果惩罚项是参数的l1范数，就是套索回归（Lasso Regrission）
'''
from sklearn.datasets import load_boston
from sklearn import linear_model

boston=load_boston()

print(boston.data[1])
clf = linear_model.Ridge(fit_intercept=False)
# clf = linear_model.Ridge()
clf.fit(boston.data,boston.target)
y_pre=clf.predict([[ 2.73100000e-02 , 0.00000000e+00 , 7.85000000e+00 , 0.00000000e+00,
   4.85000000e-01 ,  6.48100000e+00  , 7.89300000e+01  , 4.910000e+00,
   2.00000000e+00  , 3.78000000e+02 ,  1.78000000e+01 ,  3.96900000e+02,
   9.14000000e+00]])
print(y_pre)
#print(iris.target_names[y_pre])