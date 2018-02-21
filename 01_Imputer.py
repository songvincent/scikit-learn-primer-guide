import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
#x=dataset[:5,:] error:这种操作第一个方括号只能有一个切片
#x=dataset[:5][3] error:这表示这种情况也报错
#y = dataset.iloc[:,3]
'''
0     No
1    Yes
2     No
3     No
4    Yes
5    Yes
6     No
7    Yes
8     No
9    Yes
Name: Purchased, dtype: object
'''
y=dataset.iloc[:,3].values

#有无values的差别就在于一个是将列也显示出来，而另一个至显示值
#['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']

'''
查看数据
可以看出，Age 与 Salary 中都有缺失值。补全缺失值，
可以使用 Scikit-learn 库的预处理工具 Imputer。sklearn.preprocessing.Imputer，
将缺失值替换为均值、中位数、众数，下面的例子使用均值作为替换。
'''

print(X)
# print(y)
# print()

#缺失值补充

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
'''
#实例化 Imputer 类，并命名为 imputer

#missing_values：缺失值，默认为 NaN，可以为整数或者 NaN（缺失值 numpy.nan 用字符串 NaN 表示）
strategy：替换策略，字符串，默认用均值 mean 替换。
①若为 mean，用特征列的均值替换；
②若为 median，用特征列的中位数替换；
③若为 most_frequent ，用特征列的众数替换。
axis：指定轴号，如果是二维数据，默认 axis=0 代表列，axis=1 代表行。
'''

imputer.fit(X[:, 1:3])
#imputer 实例使用 fit 方法，对特征集 X 进行分析拟合。拟合后，imputer 会产生一个 statistics_ 参数，其值为 X 每列的均值、中位数、众数。
#后面的列数表示将要填充的列数
X[:, 1:3] = imputer.transform(X[:, 1:3])
#使用 imputer 的 transform 方法填充 X 的值，并重新赋值给 X。

print(dataset.iloc[:,0])
print(X)