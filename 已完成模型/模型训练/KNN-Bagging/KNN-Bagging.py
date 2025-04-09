from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import datetime

# plt.style.use('ggplot')  # 使用自带的样式进行美化
# 下面两行代码用于显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# iris = datasets.load_iris()
# x_data = iris.data[:,:2]
# y_data = iris.target

iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

x_data = iris[
    ['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
     '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
y_data = iris['state'].values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=10, test_size=0.25)
# print(iris)



# def plot(model):
    # 获取数据值所在的范围
    # x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    # y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # 生成网格矩阵
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         # np.arange(y_min, y_max, 0.02))

    # z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    # z = z.reshape(xx.shape)
    # 等高线图
    # cs = plt.contourf(xx, yy, z)

# KNN
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)

bagging_knn = BaggingClassifier(knn, random_state=10,
                                max_features=20,
                                n_estimators=50)

start_time = datetime.datetime.now()
# 输入数据建立模型
bagging_knn.fit(x_train, y_train)
end_time = datetime.datetime.now()
print("用时：" + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

# 样本散点图
# plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
# plt.show()


# bagging_tree = BaggingClassifier(dtree, n_estimators=100)
# 输入数据建立模型
# bagging_tree.fit(x_train, y_train)
# 样本散点图
# plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
# plt.show()

# np.set_printoptions(threshold=np.inf)
y_pred = bagging_knn.predict(x_test)
# print("预测值：", y_pred)
# print("测试值：", y_test)

y_pred = bagging_knn.predict_proba(x_test)[:, 1]
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
# print(bagging_tree.score(x_test, y_test))
print("准确率：", bagging_knn.score(x_test, y_test))
