import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import catboost as cb
from array import *
from sklearn.metrics import accuracy_score

# 读取 5 万行记录
data = pd.read_csv("./data/data_train.csv")
print(data.shape)  # (58191, 31)
iris = pd.read_csv(
            './data/data_train.csv')

X = iris[
    ['0.9','0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

data = data[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
             '1.6',
             '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05', 'state']]
# data[["0.95", "1", "1.05", "1.1", "1.15", "1.2", "1.25", "1.3" "1.35", "1.4", "1.45", "1.5", "1.55", "1.6",
# "1.65", "1.7", "1.75", "1.8", "1.85", "1.9", "1.95", "2", "2.05", "state"]]

# data.dropna(inplace=True)

# data["state"] = (data["state"] > 10) * 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)

cat_features_index = [0, 1]


def auc(m, train, test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))


nite = 0
max = 0
nlr = 0

for ite in range(1,40,1) :
    for lr in [0.02,0.03,0.04,0.05,0.06,1]:
        clf = cb.CatBoostClassifier(eval_metric="AUC",
                                    one_hot_max_size=31,
                                    depth=6,
                                    iterations=ite,
                                    l2_leaf_reg=1,
                                    thread_count=4,
                                    learning_rate=lr)

        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, (clf.predict(X_test)))

        if acc > max:
            max = acc
            nite = ite
            nlr = lr


print("迭代次数=", nite,nlr)
print(max)







