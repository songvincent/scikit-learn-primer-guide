#使用knn算法
'''
KNN（K-Nearest Neighbor）工作原理：
存在一个样本数据集合，也称为训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类对应的关系。
输入没有标签的数据后，将新数据中的每个特征与样本集中数据对应的特征进行比较，提取出样本集中特征最相似数据（最近邻）的分类标签。
一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k近邻算法中k的出处，通常k是不大于20的整数。最后选择k个最相似数据中出现次数最多的分类作为新数据的分类。
'''

from sklearn.datasets import load_iris  
from sklearn import neighbors  
import sklearn  
  
#查看iris数据集  
iris = load_iris()  
#print (iris)  
  
knn = neighbors.KNeighborsClassifier()  
#训练数据集  
knn.fit(iris.data, iris.target)  
#预测  
#predict = knn.predict([[0.1,0.2,0.3,0.4]])  
predict = knn.predict([[0.8,7.3,8.5,0.4]])  
print (predict)  
print (iris.target_names[predict])