from Initialize import *
from sklearn.svm import SVC

# SVC
print('Calculating SVC...')

'''
参数：
    C ：C-SVC的惩罚参数C  默认值是1.0
    kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
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
model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',
            coef0=0.0, shrinking=True, probability=False,
            tol=1e-3, cache_size=200, class_weight=None,
            verbose=False, max_iter=-1, decision_function_shape='ovr',
            random_state=None)
model.fit(train_ri, train_ril)
accuracy1 = model.score(test_ri, test_ril)

model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',
            coef0=0.0, shrinking=True, probability=False,
            tol=1e-3, cache_size=200, class_weight=None,
            verbose=False, max_iter=-1, decision_function_shape='ovr',
            random_state=None)
model.fit(train_hi, train_hil)
accuracy2 = model.score(test_hi, test_hil)

print('raw image accuracy: {:.2f}%'.format(accuracy1 * 100))
print('histogram image accuracy: {:.2f}%\n'.format(accuracy2 * 100))
