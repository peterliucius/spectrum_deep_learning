from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

iris = datasets.load_iris()
iris = pd.read_csv(
            './data/data_train.csv')

x_data = iris[
    ['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
y_data = iris['state'].values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=10, test_size=0.25)
# print(iris)

# KNN
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)

# 输入数据建立模型

best_estimators = 0
auc_max = 0
auc = 0
best_max_features = 0
for estimators in [50,100,250,500,1000]:
    for n in range(16,24,2):
        bagging_knn = BaggingClassifier(knn, random_state=10,
                                        max_features=n,
                                        n_estimators=estimators)
        bagging_knn.fit(x_train, y_train)
        auc = bagging_knn.score(x_test, y_test)

        if auc > auc_max:
            auc_max = auc
            best_estimators = estimators
            best_max_features = n

        print("运行中 auc为：", auc)

# print(bagging_tree.score(x_test, y_test))
print("best estimators：", best_estimators)
print("best_max_features：", best_max_features)
print("auc_max:", auc_max)
