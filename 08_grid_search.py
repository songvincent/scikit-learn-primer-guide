#模型选择

from sklearn import svm, grid_search, datasets
iris = datasets.load_iris()
#鸢尾花数据集，在sklearn/dataset下可见
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5,10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
print(clf)

'''
例子很简单，使用 SVM 作为预测工具。首先创建一个参数字典 parameters，
其中，核函数 kernel 包括 'linear' 与 'rbf'，惩罚参数 C 包括1与10，
然后创建 GridSearchCV 对象，将一个 SVM 回归器实例与参数字典传进去，生成一个用于网格搜索的 SVM 回归器实例，
然后对数据集进行拟合。
'''