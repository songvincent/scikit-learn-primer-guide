#谱聚类

import numpy as np
from sklearn import datasets
#生成500个个6维的数据集,聚为5类
X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
'''
n_samples: int, optional (default=100) 
The total number of points equally divided among clusters. 
待生成的样本的总数。 

n_features: int, optional (default=2) 
The number of features for each sample. 
每个样本的特征数。 

centers: int or array of shape [n_centers, n_features], optional (default=3) 
The number of centers to generate, or the fixed center locations. 
要生成的样本中心（类别）数，或者是确定的中心点。 

cluster_std: float or sequence of floats, optional (default=1.0) 
The standard deviation of the clusters. 
每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。
'''

from sklearn.cluster import SpectralClustering
# sc=SpectralClustering(n_clusters=4)
# sc.fit(X)
# y_pred=sc.predict(X)
y_pred = SpectralClustering(n_clusters=5).fit_predict(X)
print(y)
print(y_pred)
#print(len(y_pred))

from sklearn import metrics
print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred)) 