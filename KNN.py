from Initialize import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# KNN
print('Calculating KNN...')

neighbors_num = 7
'''
model = KNeighborsClassifier(n_neighbors=neighbors_num)
model.fit(train_ri, train_ril)
accuracy1 = model.score(test_ri, test_ril)

model = KNeighborsClassifier(n_neighbors=neighbors_num)
model.fit(train_hi, train_hil)
accuracy2 = model.score(test_hi, test_hil)

print('k neighbors = %d' % neighbors_num)
print('raw image accuracy: {:.2f}%'.format(accuracy1 * 100))
print('histogram image accuracy: {:.2f}%\n'.format(accuracy2 * 100))
'''

k_range = range(1, 21)
k_scores = []

for k in k_range:
    # n_neighbors 选取最近的点的个数k 默认为5
    # weight 每个点的权重 默认为'uniform'全部相等 若为'distance'则距离越近权重越大
    # algorithm 计算最近邻居的算法 默认为auto
    # leaf_size 构造树的大小，值一般选取默认值即可，太大会影响速度
    # p metric metric_params 有关minkowski_distance
    # n_jobs 默认为None 单个processor运行 若为-1则使用全部的processors
    model = KNeighborsClassifier(n_neighbors=k,
                                 weights='uniform', algorithm='auto', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=None)
    model.fit(train_ri, train_ril)
    accuracy1 = model.score(test_ri, test_ril)
    k_scores.append(accuracy1)

# 结果可视化
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('raw image accuracy')
plt.show()

k_scores = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k,
                                 weights='uniform', algorithm='auto', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=None)
    model.fit(train_hi, train_hil)
    accuracy2 = model.score(test_hi, test_hil)
    k_scores.append(accuracy2)

# 结果可视化

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('histogram image accuracy')
plt.show()
