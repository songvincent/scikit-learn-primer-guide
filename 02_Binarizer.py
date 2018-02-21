
import numpy as np

X_data = np.array([[0.3, 0.6], [0.7, 0.5]])

#Binarizer：二值化工具，将数据分为0和1，其只需要传入一个参数threshold（阈值），其小于或等于该值的数据会变为0，大于该值的数据会变为1。

from sklearn.preprocessing import Binarizer #导入库
binarizer = Binarizer(threshold=0.5) #实例化
binarizer.fit(X_data) #fit
#Binarizer(copy=True, threshold=0.5)  这表示的是一个输出

X_data = binarizer.transform(X_data) #transform
print(X_data)