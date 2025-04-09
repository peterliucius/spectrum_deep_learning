from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

X = iris[
    ['0.9', '0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)

parameters = {
    'max_iter': [50,100,250,500,1000,1500,2000]

}

clf = PLSRegression(n_components=3,max_iter=500)

best_n = 0
best_iter = 0
max_acc = 0
for n_components in [1,2,3,4,5]:
    for max_iter in [50, 100, 250, 500, 1000, 1500, 2000]:
        clf = PLSRegression(n_components=n_components, max_iter=max_iter)
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
            best_iter = max_iter
            best_n = n_components
        print('----------running----------')
        print(max_acc)


print("best_iter:", best_iter)
print("best_n:", best_n)