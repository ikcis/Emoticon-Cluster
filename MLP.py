from Initialize import *
from sklearn.neural_network import MLPClassifier

# neural network
print('Calculating neural network...')

'''
参数：
    hidden_layer_sizes 隐藏层和每层的神经元数量
    activation 激活函数
    solver 用来优化权重
    alpha 正则化项参数
    batch_size 批尺寸 有关参数更新
    learning_rate 学习率 用于权重更新
    learning_rate_init 初始学习率
    power_t 逆扩展学习率的指数
    max_iter 最大迭代次数
    shuffle 判断是否在每次迭代时对样本进行清洗
    random_state 随机数生成器的状态或种子
    tol 优化的容忍度
    verbose 是否将过程打印到stdout
    warm_start 使用之前的解决方法作为初始拟合或释放之前的解决方法
    momentum 动量梯度下降更新
    nesterovs_momentum 是否使用Nesterov’s momentum
    early_stopping 判断当验证效果不再改善的时候是否终止训练
    validation_fraction 用作早期停止验证的预留训练数据集的比例
    beta_1 估计一阶矩向量的指数衰减速率
    beta_2 估计二阶矩向量的指数衰减速率
    epsilon 数值稳定值
    n_iter_no_change 最大_没有改进_训练次数
'''

model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                      solver='lbfgs', alpha=0.0001,
                      batch_size='auto', learning_rate="constant",
                      learning_rate_init=0.001, power_t=0.5, max_iter=200,
                      shuffle=True, random_state=None, tol=1e-4,
                      verbose=False, warm_start=False, momentum=0.9,
                      nesterovs_momentum=True, early_stopping=False,
                      validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                      epsilon=1e-8, n_iter_no_change=10)
model.fit(train_ri, train_ril)
accuracy1 = model.score(test_ri, test_ril)

model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                      solver='lbfgs', alpha=0.0001,
                      batch_size='auto', learning_rate="constant",
                      learning_rate_init=0.001, power_t=0.5, max_iter=200,
                      shuffle=True, random_state=None, tol=1e-4,
                      verbose=False, warm_start=False, momentum=0.9,
                      nesterovs_momentum=True, early_stopping=False,
                      validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                      epsilon=1e-8, n_iter_no_change=10)
model.fit(train_hi, train_hil)
accuracy2 = model.score(test_hi, test_hil)
print('raw image accuracy: {:.2f}%'.format(accuracy1 * 100))
print('histogram image accuracy: {:.2f}%\n'.format(accuracy2 * 100))
