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
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9']].values
Y = iris['state'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

# 参数
params = {
    'learning_rate': 0.4,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 9,
    'objective': 'multiclass',  # 目标函数
    'num_class': 3,
    'num_boost_round': 42,
    'min_child_samples': 24,
    'num_leaves': 25
}

start_time = datetime.datetime.now()
# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data])

end_time = datetime.datetime.now()
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

# 模型预测
y_pred = gbm.predict(X_test)
n_rows = y_pred.shape[0]
y_pred = [list(x).index(max(x)) for x in y_pred]



# print(y_pred)
# print(y_test)

# 模型评估
tp = 0
tn = 0
fp = 0
fn = 0
for k in range(0, n_rows):
    if y_pred[k] == 1 and y_test[k] == 1:
        tp = tp + 1
    elif y_pred[k] == 0 and y_test[k] == 0:
        tn = tn + 1
    elif y_pred[k] == 1 and y_test[k] == 0:
        fp = fp + 1
    elif y_pred[k] == 0 and y_test[k] == 1:
        fn = fn + 1
    else:
        print('错误分类样本的序号：', k + 1)


print("精确率：", tp/(tp + fp))
print("召回率：", tp/(tp + fn))
print("负类召回率：", tn/(tn + fp))
print("Accuracy:", (tp+tn)/(tp+tn+fn+fp))





