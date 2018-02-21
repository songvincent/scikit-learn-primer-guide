'''
分类是对指定的对象进行分类，
Scikit-learn 实现的分类算法包括支持向量机（SVM）、最近邻（kNN）、逻辑回归（Logistic Regression）、随机森林（RF）和决策树（Decision Tree）等等。
'''

'''
分类套路如下：

导入库
实例化分类/回归器
拟合数据集（fit）
预测测试集（predict）
'''
# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


#这里相当于对数据集进行了shuffle后按照给定的test_size 进行数据集划分。
#将训练集划分为两部分，size表示test的比例
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#标准化，可以直接用训练集去初始化测试集
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Support Vector Regressor（支持向量回归器）
#Support Vector Classifier（支持向量分类器）
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
print(X_train.shape)
'''
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,

tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)

参数：
SVC的参数非常齐全，主要有：

C：float，可选（默认为1.0），是 SVM 种的惩罚参数。
kernel：string，可选（默认为'rbf'），是核函数策略，可以设为 'linear'（线性SVM），主要用于线性可分的情况，参数少，速度快。
'poly'：多项式核函数，比线性 SVM 的参数更多。
'rbf'：径向基核函数，通常用于线性不可分的情况，参数多，比较耗时，需要调参 。
'sigmoid'：Sigmod核函数。
degree：int。可选（默认为3），多项式核函数的阶数，使用其他核函数可无视。

***********************************
C：C-SVC的惩罚参数C?默认值是1.0

  C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
  C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 

  0 – 线性：u'v

  1 – 多项式：(gamma*u'*v + coef0)^degree

  2 – RBF函数：exp(-gamma|u-v|^2)

  3 –sigmoid：tanh(gamma*u'*v + coef0)

degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。

gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features

coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

probability ：是否采用概率估计？.默认为False

shrinking ：是否采用shrinking heuristic方法，默认为true

tol ：停止训练的误差值大小，默认为1e-3

cache_size ：核函数cache缓存大小，默认为200

class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)

verbose ：允许冗余输出？

max_iter ：最大迭代次数。-1为无限制。

decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3

random_state ：数据洗牌时的种子值，int值
'''

y_pred = classifier.predict(X_test)

# print("prediction:",y_pred)
# print("test：",y_test)

#sklearn.metrics中的评估方法
#评估方法介绍网址：http://blog.csdn.net/cherdw/article/details/55813071
#cofusion_matrix 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#print("cm",cm)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# print('********')
# print(X1)
# print('******')
# print(X2)
#meshgrid 生成两个矩阵X和Y，行数与Y元素数量相同
#X 为x的多次重复，Y 为同样的数字序列的递增
#X列同行不同，Y行同列不同

#网址：https://www.cnblogs.com/huanggen/p/7533088.html
#简单介绍了几种图形的简单绘制，散点图，柱状图，等高线图，图片和3D数据

#matplotlib教程：http://blog.csdn.net/sunshine_in_moon/article/details/46573117

#alpha表示透明度
#cmap表示渐变标准，颜色填充
#添加colormap的对象是灰度图，可以变成热量图，从而更加明显的发现一些规律，适用于一些雷达图像等

print(X1.shape)
print(classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape).shape)
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             # alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#contourf此命令用来填充二维等值线图，即先画出不同等值线，然后将相邻的等值线之间用同一颜色进行填充，填充用的颜色决定于当前的色图颜色
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape((592,-1)),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#numpy.ravel() 多维降为一维
#转置以后，一个X1对应一个x2
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plt.show()
#以上为产生等高线，区别0，1
#contourf(X,Y,Z):画出矩阵Z的等值线图，其中X与Y用于指定x轴与y轴的范围，若X与Y为矩阵，则必须与Z同型；若X或Y有不规则的间距，contourf(X,Y,Z)

'''
enumerate()是python的内置函数
enumerate在字典上是枚举、列举的意思
对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
enumerate多用于在for循环中得到计数
例如对于一个seq，得到：
(0, seq[0]), (1, seq[1]), (2, seq[2])
'''

# print(len(X_set))
# print(len(X_set[y_set==0,0]))

for i, j in enumerate(np.unique(y_set)):
    # print(j)
    # print(y_set==j)
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
	# plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                # c = ListedColormap(('red', 'green'))(i))
				
#ListedColormap(('red', 'green','blue'))(i)
#后面的i表示的是选择哪个颜色

plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
#显示图例
plt.legend()
plt.show()