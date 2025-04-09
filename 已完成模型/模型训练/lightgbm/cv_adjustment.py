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


# 模型训练
gbm = lgb.LGBMClassifier(
    learning_rate=0.3,
    lambda_l1=0.1,
    lambda_l2=0.2,
    max_depth=9,
    objective='regression',  # 目标函数
    )

gbm.fit(X_train, y_train)
# 模型预测
y_pred = gbm.predict(X_test)


# 模型评估
print(accuracy_score(y_test, y_pred))


for num_leaves in range(5, 100, 5):
    for max_depth in range(3, 8, 1):

        for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
                for bagging_freq in range(0, 50, 5):

                    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
                            for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
