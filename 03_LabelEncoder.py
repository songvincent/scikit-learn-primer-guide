#LabelEncoder：标签编码工具，将标签编码为数字，其顺序排列基于 ASCII 码。比如，将 “amsterdam”、“paris”、“tokyo” 分别标记为0、1、2。
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(["paris", "paris", "tokyo", "amsterdam"])

print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))

'''
其余预处理工具，如OneHotEncoder、MaxAbsScaler、PolynomialFeatures 等等，
使用方法都与此相同，仅仅是传入的参数有差异。
'''