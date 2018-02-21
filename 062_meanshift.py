#均值漂移
#不需要预先制定类数
#指定类数时需要用到bandwidth
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn import datasets 
  
# Generate sample data 创建数据集  

X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.6, 0.4], random_state=11)
#X, y = datasets.make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)  
  
  
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)  

#ms = MeanShift(n_clusters=5)  
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)  
ms.fit(X)  
labels = ms.labels_  
print(labels)
cluster_centers = ms.cluster_centers_  
  
labels_unique = np.unique(labels)  
n_clusters_ = len(labels_unique)  
  
print ("number of estimated clusters: %d" % n_clusters_) 

print(ms.predict(X))