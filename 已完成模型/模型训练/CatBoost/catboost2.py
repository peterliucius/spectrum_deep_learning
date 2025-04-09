import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import catboost as cb
from array import *
from sklearn.metrics import accuracy_score
import datetime

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

# With Categorical features
clf = cb.CatBoostClassifier(eval_metric="AUC",
                            one_hot_max_size=31,
                            depth=3,
                            iterations=186,
                            l2_leaf_reg=1,
                            thread_count=4,
                            learning_rate=0.9)

params = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
          'iterations': [250, 100, 500, 1000],
          'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
          'l2_leaf_reg': [3, 1, 5, 10, 100],
          'border_count': [32, 5, 10, 20, 50, 100, 200],
          'ctr_border_count': [50, 5, 10, 20, 100, 200],
          'thread_count': 4}

start_time = datetime.datetime.now()  # 计时开始
clf.fit(X_train, y_train)
end_time = datetime.datetime.now()  # 计时结束
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

y_pred = clf.predict_proba(X_test)[:, 1]
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


np.set_printoptions(threshold=np.inf)


# 输出测试集结果
# print("真实值：", y_test)
# 输出准确率
# print("(训练集准确率，测试集准确率)：", auc(clf, X_train, X_test))

# 计算预测值
y_pred = clf.predict(X_test)

# 输出预测值
# print("预测值：", y_pred)

print('准确率：', (tp+tn)/(tp+tn+fp+fn))


