from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import numpy as np
import datetime

iris = datasets.load_iris()
iris = pd.read_csv(
            './data/data_train.csv')

X = iris[
    ['0.9', '0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)  # 划分数据集

print('----------rbf----------')
parameters = {
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'C': [1,5,10,20,50,100,150],
    'epsilon':[0.01,0.05,0.1,0.5]
}

max_acc = 0
max_gamma = 0
max_C = 0
max_epsilon = 0
for gamma in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    for C in [1, 5, 10, 20, 50, 100, 150]:
        for epsilon in [0.01, 0.05, 0.1, 0.5]:
            svr_rbf = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
            clf = svr_rbf
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
                max_acc = acc
                max_C = C
                max_gamma = gamma
                max_epsilon = epsilon
            print(max_acc)


print("max_gamma:", max_gamma)
print("max_C", max_C)
print("max_epsilon", max_epsilon)
print(max_acc)

print('----------running----------')
