from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import pandas as pd

iris = pd.read_csv(
            './data/data_train.csv')

X = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.9', '1.95', '2', '2.05']]
Y = iris['state']
# 把数据分为训练数据集和测试数据集(20%数据作为测试数据集）
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model = ElasticNet(alpha=0.00001,l1_ratio=0.1, normalize=True)
model.fit(X_train, y_train)
# 查看模型的斜率sparse_coef_ 是从coef_ 导出的只读属性
print(model.coef_)
print(model.sparse_coef_)
# 查看模型的截距
print(model.intercept_)

train_score = model.score(X_train, y_train)  # 模型对训练样本得准确性
test_score = model.score(X_test, y_test)  # 模型对测试集的准确性
print(train_score)
print(test_score)

model = ElasticNetCV(alphas=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001], l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
model.fit(X_train, y_train)
# 查看模型的斜率
print(model.coef_)
# 查看模型的截距
print(model.intercept_)
train_score = model.score(X_train, y_train)  # 模型对训练样本得准确性
test_score = model.score(X_test, y_test)  # 模型对测试集的准确性
print(train_score)
print(test_score)
# 最优alpha
print(model.alpha_)
print(model.l1_ratio_)
