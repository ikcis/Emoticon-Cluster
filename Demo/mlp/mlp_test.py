from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
import gzip

# 加载数据
# mnist = fetch_mldata("MNIST original")
with gzip.open('mnist.pkl.gz') as fp:
    training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
x_training_data, y_training_data = training_data
x_valid_data, y_valid_data = valid_data
x_test_data, y_test_data = test_data
classes = np.unique(y_test_data)

# 将验证集和训练集合并
x_training_data_final = np.vstack((x_training_data, x_valid_data))
y_training_data_final = np.append(y_training_data, y_valid_data)

'''
# 设置神经网络模型参数
# mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
# 使用solver='lbfgs',准确率为79%，比较适合小(少于几千)数据集来说，且使用的是全训练集训练，比较消耗内存
# mlp = MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
# 使用solver='adam'，准确率只有67%
'''
mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                    max_iter=10, verbose=10, learning_rate_init=.1)
# 使用solver='sgd'，准确率为98%，且每次训练都会分batch，消耗更小的内存

# 训练模型
mlp.fit(x_training_data_final, y_training_data_final)

# 查看模型结果
mlp.score(x_test_data, y_test_data)
