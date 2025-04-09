import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import datetime

iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

X = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.85', '1.9', '1.95', '2',
       '2.05']].values
Y = iris['state'].values
np.set_printoptions(threshold=np.inf)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)  # 分割测试集与训练集


rf_clf = ExtraTreesClassifier(n_estimators=1000,
                              random_state=10,
                              max_depth=None,
                              oob_score=True,
                              bootstrap=True)


start_time = datetime.datetime.now()  # 计时开始
rf_clf.fit(X_train, y_train)  # 模型拟合
end_time = datetime.datetime.now()  # 计时结束
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

acc = accuracy_score(y_test, (rf_clf.predict(X_test)))

print(acc)
y_pred = rf_clf.predict_proba(X_test)[:, 1]
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






