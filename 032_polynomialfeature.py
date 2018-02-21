'''
在建模过程中多次用到过sklearn.preprocessing.PolynomialFeatures，可以理解为专门生成多项式特征，并且多项式包含的是相互影响的特征集，
比如：一个输入样本是２维的。形式如[a,b] ,则二阶多项式的特征集如下[1,a,b,a^2,ab,b^2]。

官网文档：http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

参数：

degree : integer，多项式阶数，默认为2；

interaction_only : boolean, default = False，如果值为true(默认是false),则会产生相互影响的特征集；

include_bias : boolean，是否包含偏差列。
'''
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)  
print(X)

poly = PolynomialFeatures(2) #设置多项式阶数为２，其他的默认  
m=poly.fit_transform(X)  

poly = PolynomialFeatures(interaction_only=True)# 默认的阶数是２，同时设置交互关系为true  
n=poly.fit_transform(X)  

print(m)
print(n)