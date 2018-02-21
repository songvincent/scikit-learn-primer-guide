#SVR
#支持向量回归
from __future__ import division  
import time  
import numpy as np  
from sklearn.svm import SVR  
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import learning_curve  
import matplotlib.pyplot as plt  
  
rng = np.random.RandomState(0)  
  
#http://blog.csdn.net/lanchunhui/article/details/50405670
#关于random的一系列接口解释
#############################################################################  
# 生成随机数据  
X = 5 * rng.rand(10000, 1)
#rand方法生成（0，1）之间的随机数字
#括号内的内容为生成矩阵的形式
y = np.sin(X).ravel()  

# print(X)
# print(y)
# print(len(X))

# 在标签中对每50个结果标签添加噪声  
#print(X.shape[0])
y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0]/50)))  
#意思是每隔50个元素取一个元素
#print(len(y[::50]))
#print(y[::50])

X_plot = np.linspace(0, 5, 100000)[:, None]  
#后面的方括号这是一种比较复杂的切片方式，即将每一个单独的元素拿出来不做任何处理，直接作为元组的一个新的部分
#简单理解就是将一维的数组变为二维，二维可以变为三维
#X_plot = np.linspace(0, 5, 100000)
#print(X_plot)
#linspace接口创建等差数列。
#前两个参数分别是数列的开头与结尾。如果写入第三个参数，可以制定数列的元素个数。
  
#############################################################################  
# 训练SVR模型  

#训练规模  
train_size = 100  
#初始化SVR  
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,  
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],  
                               "gamma": np.logspace(-2, 2, 5)})  
#记录训练时间  
t0 = time.time()  
#训练  
svr.fit(X[:train_size], y[:train_size])  
svr_fit = time.time() - t0  
  
t0 = time.time()  
#测试  
y_svr = svr.predict(X_plot)  
svr_predict = time.time() - t0  
print(svr_fit,svr_predict)

#############################################################################  
# 对结果进行显示  
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1)  
plt.hold('on')  
plt.plot(X_plot, y_svr, c='r',  
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))  
  
plt.xlabel('data')  
plt.ylabel('target')  
plt.title('SVR versus Kernel Ridge')  
plt.legend()  
  
#plt.figure()  
plt.show()

# 对学习过程进行可视化  
plt.figure()  
#重新建立一个绘图窗口
svr = SVR(kernel='rbf', C=1e1, gamma=0.1)  
train_sizes, train_scores_svr, test_scores_svr =\
    learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),  
                   scoring="neg_mean_squared_error", cv=10)  

'''
estimator：所使用的分类器

X:array-like, shape (n_samples, n_features)

 训练向量，n_samples是样本的数量，n_features是特征的数量

y:array-like, shape (n_samples) or (n_samples, n_features), optional

目标相对于X分类或者回归

train_sizes:array-like, shape (n_ticks,), dtype float or int

训练样本的相对的或绝对的数字，这些量的样本将会生成learning curve。如果dtype是float，他将会被视为最大数量训练集的一部分（这个由所选择的验证方法所决定）。否则，他将会被视为训练集的绝对尺寸。要注意的是，对于分类而言，样本的大小必须要充分大，达到对于每一个分类都至少包含一个样本的情况。

cv:int, cross-validation generator or an iterable, optional

确定交叉验证的分离策略

--None，使用默认的3-fold cross-validation,

--integer,确定是几折交叉验证

--一个作为交叉验证生成器的对象

--一个被应用于训练/测试分离的迭代器

verbose : integer, optional

控制冗余：越高，有越多的信息
.....................................................
返回值：
train_sizes_abs：array, shape = (n_unique_ticks,), dtype int

用于生成learning curve的训练集的样本数。由于重复的输入将会被删除，所以ticks可能会少于n_ticks.

train_scores : array, shape (n_ticks, n_cv_folds)

在训练集上的分数

test_scores : array, shape (n_ticks, n_cv_folds)

在测试集上的分数
'''
plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",  
         label="SVR")  
#o- 显示出相应的圆点
plt.xlabel("Train size")  
plt.ylabel("Mean Squared Error")  
plt.title('Learning curves')  
plt.legend(loc="best")  
  
plt.show()  