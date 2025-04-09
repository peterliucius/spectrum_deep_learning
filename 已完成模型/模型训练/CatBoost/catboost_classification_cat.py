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
    ['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

# data[["0.95", "1", "1.05", "1.1", "1.15", "1.2", "1.25", "1.3" "1.35", "1.4", "1.45", "1.5", "1.55", "1.6",
# "1.65", "1.7", "1.75", "1.8", "1.85", "1.9", "1.95", "2", "2.05", "state"]]

# data.dropna(inplace=True)

# data["state"] = (data["state"] > 10) * 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)

cat_features_index = [0, 1]


def auc(m, train, test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))


ndep = 0
nite = 0
nlr = 0
nl2 = 0
max = 0

for dep in [1, 2, 3, 6, 4, 5, 7, 8, 9, 10]:
    for ite in [250, 100, 500, 1000, 2000]:
        for lr in [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]:
            for l2 in [3, 1, 5, 10, 100]:
                clf = cb.CatBoostClassifier(eval_metric="AUC",
                                            one_hot_max_size=31,
                                            depth=dep,
                                            iterations=ite,
                                            l2_leaf_reg=l2,
                                            learning_rate=lr,
                                            thread_count=4)

                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, (clf.predict(X_test)))

                if acc > max:
                    max = acc
                    ndep = dep
                    nite = ite
                    nl2 = l2
                    nlr = lr


print("dep,ite,lr,l2=", ndep, nite, nlr, nl2)
print(max)







