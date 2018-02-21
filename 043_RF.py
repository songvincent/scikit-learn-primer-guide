'''
随机森林
from sklearn.ensemble import RandomForestClassifier 
RF = RandomForestClassifier() 
RF = RF.fit(train_x, train_y) 
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris=load_iris()

RF = RandomForestClassifier()
rf = RF.fit(iris.data,iris.target)

y_pre=rf.predict([[1,2,3,4]])

print(iris.target_names[y_pre])