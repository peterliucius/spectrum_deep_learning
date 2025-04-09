import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import datetime

# 设置模型参数
params = {
    'max_depth': 15,
    'learning_rate': 0.01,
    'n_estimators': 2000,
    'min_child_weight': 2,
    'reg_alpha':  1,
    'reg_lambda': 1
}

# 参数定义
plst = list(params.items())

iris = datasets.load_iris()
iris = pd.read_csv(
            './data/data_train.csv')

X = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
          '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

xlf = xgb.XGBClassifier(max_depth=2,
                        learning_rate=0.1,
                        n_estimators=1500,
                        verbosity=0,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=2,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0.1,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440)

start_time = datetime.datetime.now()
# 对测试集进行预测
xlf.fit(X_train, y_train)  # 模型拟合
end_time = datetime.datetime.now()
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

y_pred = xlf.predict(X_test)  # 计算预测值

np.set_printoptions(threshold=np.inf)  # 使数据数据不省略
# 输出预测值
# print('预测值：', y_pred)

# 开始计算tp等值
y_pred = xlf.predict_proba(X_test)[:, 1]
n_rows = y_pred.shape[0]
# 预测值结果存储
output = np.empty(shape=(n_rows, ), dtype=int)

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


print("精确率：", tp/(tp + fp))
print("召回率：", tp/(tp + fn))
print("负类召回率：", tn/(tn + fp))


# 计算准确率
print("Accuracy:", (tp+tn)/(tp+tn+fn+fp))

# 绘制特征重要性
# plot_importance(model)
# plt.show();
