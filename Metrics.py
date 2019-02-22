from Initialize import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.svm import SVC

neighbors_num = 7

k_range = range(4, 5)

k_scores = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k,
                                 weights='uniform', algorithm='auto', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=None)
    model.fit(train_hi, train_hil)
    accuracy2 = model.score(test_hi, test_hil)
    k_scores.append(accuracy2)
    y_true = test_hil
    y_pred = model.predict(test_hi)
    cr = classification_report(y_true, y_pred)
    print(cr)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

"""
              precision    recall  f1-score   support

         Ali       0.60      0.75      0.67        12
     Cangshu       0.76      0.86      0.81        22
       Huaji       0.78      0.78      0.78        23
       Panda       0.61      0.94      0.74        18
     Sadfrog       1.00      0.36      0.53        25

   micro avg       0.72      0.72      0.72       100
   macro avg       0.75      0.74      0.71       100
weighted avg       0.78      0.72      0.70       100

[[ 9  1  0  2  0]
 [ 0 19  1  2  0]
 [ 1  1 18  3  0]
 [ 0  0  1 17  0]
 [ 5  4  3  4  9]]
 """
