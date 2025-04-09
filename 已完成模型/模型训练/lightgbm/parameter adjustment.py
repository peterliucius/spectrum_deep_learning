import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import sys
import datetime

np.set_printoptions(threshold=np.inf)
# 加载数据
iris = datasets.load_iris()
iris = pd.read_csv(
            './data/data_train.csv')

X = iris[
    ['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
             '1.6',
             '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

auc_max = 0
n1 = 0
n2 = 0
n3 = 0
n4 = 0
n5 = 0

for nl in range(20, 60, 5):
    for md in range(6, 20, 1):
        for lr in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 1]:
            for m_samples in [22, 23, 24, 25]:
                for m_weight in [None, 0.001, 0.002]:
                    params = {
                        'num_leaves': nl,
                        'learning_rate': lr,
                        'lambda_l1': 0.1,
                        'lambda_l2': 0.2,
                        'max_depth': md,
                        'objective': 'multiclass',  # 目标函数
                        'min_child_samples': m_samples,
                        'min_child_weight': m_weight,
                        'num_class': 3,
                    }
                    gbm = lgb.train(params, train_data, valid_sets=[validation_data])

                    # 模型预测
                    y_pred = gbm.predict(X_test)
                    y_pred = [list(x).index(max(x)) for x in y_pred]
                    auc = accuracy_score(y_test, y_pred)
                    if auc > auc_max:
                        auc_max = auc
                        n1 = nl
                        n2 = lr
                        n3 = md
                        n4 = m_weight
                        n5 = m_samples
                    print(auc_max)

print("num_leaves ", n1)
print("learning_rate ", n2)
print("max_depth ", n3)
print("min_child_weight ", n4)
print("min_child_samples ", n5)
print(auc_max)


