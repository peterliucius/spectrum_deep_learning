from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import numpy as np
from sklearn.metrics import r2_score

iris = datasets.load_iris()
iris = pd.read_csv(
            './data/data_train.csv')

X = iris[
    ['0.9','0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)  # 划分数据集

parameters = {
    'learning_rate': [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'n_estimators': [50,100,250,500,1000,1500,2000]

}

best_lr = 0
best_ne = 0
max_acc = 0
for lr in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    for ne in [50,100,250,500,1000,1500,2000]:
        clf = AdaBoostClassifier(n_estimators=ne,
                                 learning_rate=lr,
                                 algorithm='SAMME',
                                 random_state=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        n_rows = y_pred.shape[0]
        # 预测值结果存储
        output = np.empty(shape=(n_rows,), dtype=int)

        for i in range(n_rows):
            if y_pred[i] > 0.5:
                output[i] = 1
            else:
                output[i] = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for k in range(0, n_rows):
            if output[k] == 1 and y_test[k] == 1:
                tp = tp + 1
            elif output[k] == 0 and y_test[k] == 0:
                tn = tn + 1
            elif output[k] == 1 and y_test[k] == 0:
                fp = fp + 1
            elif output[k] == 0 and y_test[k] == 1:
                fn = fn + 1
            else:
                print('错误分类样本的序号：', k + 1)

        acc = (tp + tn) / (tp + tn + fn + fp)
        if acc > max_acc:
            best_lr = lr
            best_ne = ne
            max_acc = acc
        print(acc)

print("best_ne:", best_ne)
print("best_lr", best_lr)
print(max_acc)
print('----------SAMME running----------')

best_lr = 0
best_ne = 0
max_acc = 0
for lr in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    for ne in [50,100,250,500,1000,1500,2000]:
        clf = AdaBoostClassifier(n_estimators=ne,
                                 learning_rate=lr,
                                 algorithm='SAMME.R',
                                 random_state=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        n_rows = y_pred.shape[0]
        # 预测值结果存储
        output = np.empty(shape=(n_rows,), dtype=int)

        for i in range(n_rows):
            if y_pred[i] > 0.5:
                output[i] = 1
            else:
                output[i] = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for k in range(0, n_rows):
            if output[k] == 1 and y_test[k] == 1:
                tp = tp + 1
            elif output[k] == 0 and y_test[k] == 0:
                tn = tn + 1
            elif output[k] == 1 and y_test[k] == 0:
                fp = fp + 1
            elif output[k] == 0 and y_test[k] == 1:
                fn = fn + 1
            else:
                print('错误分类样本的序号：', k + 1)

        acc = (tp + tn) / (tp + tn + fn + fp)
        if acc > max_acc:
            best_lr = lr
            best_ne = ne
            max_acc = acc

print("best_ne:", best_ne)
print("best_lr", best_lr)
print(max_acc)
print('----------SAMME.R running----------')

