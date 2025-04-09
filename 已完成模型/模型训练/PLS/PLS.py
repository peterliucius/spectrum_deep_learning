from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import datetime
from sklearn.metrics import r2_score

iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

X = iris[
    ['0.9', '0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
Y = iris['state'].values
start_time = datetime.datetime.now()  # 计时开始
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

clf = PLSRegression(n_components=5, max_iter=50,)

clf.fit(X_train, y_train)
end_time = datetime.datetime.now()  # 计时结束

np.set_printoptions(threshold=np.inf)  # 使数据数据不省略
# 输出预测值
# print('预测值：', y_pred)

y_pred = clf.predict(X_test)
n_rows = y_pred.shape[0]
# 预测值结果存储
output = np.empty(shape=(n_rows, ), dtype=int)

for i in range(n_rows):
    if y_pred[i] >= 0.5:
        output[i] = 1
    else:
        output[i] = 0
print(output)
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

print("r2 score:", r2_score(y_test, y_pred))
print("r2 score:", r2_score(y_test, output))
print("精确率：", tp/(tp + fp))
print("召回率：", tp/(tp + fn))
print("负类召回率：", tn/(tn + fp))
# 计算准确率
print("Accuracy:", (tp+tn)/(tp+tn+fn+fp))
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')