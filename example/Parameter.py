from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

KNeighborsClassifier(n_neighbors=k,
                     weights='uniform', algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None)

'''

n_neighbors : int, 可选参数(默认为 5)
    用于kneighbors查询的默认邻居的数量
weights（权重） : str or callable(自定义类型), 可选参数(默认为 ‘uniform’)
    用于预测的权重函数。可选参数如下:
        ‘uniform’ : 统一的权重. 在每一个邻居区域里的点的权重都是一样的。
        ‘distance’ : 权重点等于他们距离的倒数。使用此函数，更近的邻居对于所预测的点的影响更大。
        [callable] : 一个用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组。
algorithm（算法） : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, 可选参数（默认为 'auto'）
    计算最近邻居用的算法：
        ‘ball_tree’ 使用算法 BallTree
        ‘kd_tree’ 使用算法 KDTree
        ‘brute’ 使用暴力搜索.
        ‘auto’ 会基于传入fit方法的内容，选择最合适的算法。
    注意: 如果传入fit方法的输入是稀疏的，将会重载参数设置，直接使用暴力搜索。
leaf_size（叶子数量） : int, 可选参数(默认为 30)
    传入BallTree或者KDTree算法的叶子数量。此参数会影响构建、查询BallTree或者KDTree的速度，以及存储BallTree或者KDTree所需要的内存大小。 此可选参数根据是否是问题所需选择性使用。
p : integer, 可选参数(默认为 2)
    用于Minkowski metric（闵可夫斯基空间）的超参数。p = 1, 相当于使用曼哈顿距离 (l1)，p = 2, 相当于使用欧几里得距离(l2)  对于任何 p ，使用的是闵可夫斯基空间(l_p)
metric（矩阵） : string or callable, 默认为 ‘minkowski’
    用于树的距离矩阵。默认为闵可夫斯基空间 ，如果和p=2一块使用相当于使用标准欧几里得矩阵. 所有可用的矩阵列表请查询 DistanceMetric 的文档。
metric_params（矩阵参数） : dict, 可选参数(默认为 None)
    给矩阵方法使用的其他的关键词参数。
n_jobs : int, 可选参数(默认为 1)
    用于搜索邻居的，可并行运行的任务数量。如果为 -1, 任务数量设置为CPU核的数量。不会影响 fit 方法。

'''

SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',
    coef0=0.0, shrinking=True, probability=False,
    tol=1e-3, cache_size=200, class_weight=None,
    verbose=False, max_iter=-1, decision_function_shape='ovr',
    random_state=None)

'''

经常用到sklearn中的 svm.SVC 函数，这个函数也是基于libsvm实现的，所以在参数设置上有很多相似的地方。（libsvm中的二次规划问题的解决算法是SMO）

参数：
    C：C-SVC的惩罚参数C  默认值是1.0 
        C越大，越惩罚 松弛变量（误分类），希望 松弛变量（误分类） 接近0，趋向于对训练集全分对的情况，对训练集测试时准确率很高，但泛化能力弱。
            【泛化能力(generalization ability)是指机器学习算法对新鲜样本的适应能力。学习的目的是学到隐含在数据对背后的规律，
            对具有同一规律的学习集以外的数据，经过训练的网络也能给出合适的输出，该能力称为泛化能力。】
        C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
        
    kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
      　　0 – 线性：u'v
     　　 1 – 多项式：(gamma*u'*v + coef0)^degree
      　　2 – RBF函数：exp(-gamma|u-v|^2)
      　　3 –sigmoid：tanh(gamma*u'*v + coef0)
   degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
   gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
   coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
   probability ：是否采用概率估计？.默认为False
   shrinking ：是否采用shrinking heuristic方法，默认为true
   tol ：停止训练的误差值大小，默认为1e-3
   cache_size ：核函数cache缓存大小，默认为200
   class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
   verbose ：允许冗余输出？
   max_iter ：最大迭代次数。-1为无限制。
   decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
   random_state ：数据洗牌时的种子值，int值
   
主要调节的参数有：C、kernel、degree、gamma、coef0。

'''

MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
              solver='lbfgs', alpha=0.0001,
              batch_size='auto', learning_rate="constant",
              learning_rate_init=0.001, power_t=0.5, max_iter=200,
              shuffle=True, random_state=None, tol=1e-4,
              verbose=False, warm_start=False, momentum=0.9,
              nesterovs_momentum=True, early_stopping=False,
              validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
              epsilon=1e-8, n_iter_no_change=10)

'''

参数说明: 
1. hidden_layer_sizes :例如hidden_layer_sizes=(50, 50)，表示有两层隐藏层，第一层隐藏层有50个神经元，第二层也有50个神经元。 
2. activation :激活函数,{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, 默认relu 
- identity：f(x) = x 
- logistic：其实就是sigmod,f(x) = 1 / (1 + exp(-x)). 
- tanh：f(x) = tanh(x). 
- relu：f(x) = max(0, x) 
3. solver： {‘lbfgs’, ‘sgd’, ‘adam’}, 默认adam，用来优化权重 
- lbfgs：quasi-Newton方法的优化器 
- sgd：随机梯度下降 
- adam： Kingma, Diederik, and Jimmy Ba提出的机遇随机梯度的优化器 
注意：默认solver ‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好。 
4. alpha :float,可选的，默认0.0001,正则化项参数 
5. batch_size : int , 可选的，默认’auto’,随机优化的minibatches的大小batch_size=min(200,n_samples)，如果solver是’lbfgs’，分类器将不使用minibatch 
6. learning_rate :学习率,用于权重更新,只有当solver为’sgd’时使用，{‘constant’，’invscaling’, ‘adaptive’},默认constant 
- ‘constant’: 有’learning_rate_init’给定的恒定学习率 
- ‘incscaling’：随着时间t使用’power_t’的逆标度指数不断降低学习率learning_rate_ ，effective_learning_rate = learning_rate_init / pow(t, power_t) 
- ‘adaptive’：只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少tol时，将当前学习率除以5. 
7. power_t: double, 可选, default 0.5，只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。 
8. max_iter: int，可选，默认200，最大迭代次数。 
9. random_state:int 或RandomState，可选，默认None，随机数生成器的状态或种子。 
10. shuffle: bool，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。 
11. tol：float, 可选，默认1e-4，优化的容忍度 
12. learning_rate_int:double,可选，默认0.001，初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。 
14. verbose : bool, 可选, 默认False,是否将过程打印到stdout 
15. warm_start : bool, 可选, 默认False,当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。 
16. momentum : float, 默认 0.9,动量梯度下降更新，设置的范围应该0.0-1.0. 只有solver=’sgd’时使用. 
17. nesterovs_momentum : boolean, 默认True, Whether to use Nesterov’s momentum. 只有solver=’sgd’并且momentum > 0使用. 
18. early_stopping : bool, 默认False,只有solver=’sgd’或者’adam’时有效,判断当验证效果不再改善的时候是否终止训练，当为True时，自动选出10%的训练数据用于验证并在两步连续迭代改善，低于tol时终止训练。 
19. validation_fraction : float, 可选, 默认 0.1,用作早期停止验证的预留训练数据集的比例，早0-1之间，只当early_stopping=True有用 
20. beta_1 : float, 可选, 默认0.9，只有solver=’adam’时使用，估计一阶矩向量的指数衰减速率，[0,1)之间 
21. beta_2 : float, 可选, 默认0.999,只有solver=’adam’时使用估计二阶矩向量的指数衰减速率[0,1)之间 
22. epsilon : float, 可选, 默认1e-8,只有solver=’adam’时使用数值稳定值。 

属性说明： 
- classes_:每个输出的类标签 
- loss_:损失函数计算出来的当前损失值 
- coefs_:列表中的第i个元素表示i层的权重矩阵 
- intercepts_:列表中第i个元素代表i+1层的偏差向量 
- n_iter_ ：迭代次数 
- n_layers_:层数 
- n_outputs_:输出的个数 
- out_activation_:输出激活函数的名称。

'''
