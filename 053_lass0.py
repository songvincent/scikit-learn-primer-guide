
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso

boston=load_boston()
lasso = Lasso()
lasso.fit(boston.data,boston.target)
y_pre=lasso.predict([[ 2.73100000e-02 , 0.00000000e+00 , 7.85000000e+00 , 0.00000000e+00,
   4.85000000e-01 ,  6.48100000e+00  , 7.89300000e+01  , 4.910000e+00,
   2.00000000e+00  , 3.78000000e+02 ,  1.78000000e+01 ,  3.96900000e+02,
   9.14000000e+00]])
print(y_pre)