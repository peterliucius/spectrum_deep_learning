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
_ne = 0
_ni = 0
for num_leaves in range(20,30,1):
    for num_boost_round in range(2,70,2):
        params = {
            'num_leaves': num_leaves,
            'learning_rate': 0.4,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 9,
            'num_boost_round': num_boost_round,
            'objective': 'multiclass',  # 目标函数
            'min_child_samples': 24,
            'min_child_weight': None,
            'num_class': 3,
        }
        gbm = lgb.train(params, train_data, valid_sets=[validation_data])

        # 模型预测
        y_pred = gbm.predict(X_test)
        y_pred = [list(x).index(max(x)) for x in y_pred]
        auc = accuracy_score(y_test, y_pred)
        if auc > auc_max:
            auc_max = auc
            _ne = num_leaves
            _ni = num_boost_round

print(_ne)
print(_ni)
print(auc_max)
