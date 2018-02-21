#决策树

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

y_pre = clf.predict([[1,2,3,4],
					[1.2,2.3,3.4,6.7]])
print(iris.target_names[y_pre])