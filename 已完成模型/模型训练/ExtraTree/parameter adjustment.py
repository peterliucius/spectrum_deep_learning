import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

X = iris[['0.9', '0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35',
       '1.45', '1.55']]
Y = iris['state']  # 不能用.values,会报错
np.set_printoptions(threshold=np.inf)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)  # 分割测试集与训练集

n_dep = 0
m_estimators = 0
n_w_leaf = 0
n_s_leaf = 0
n_max = 0
for estimators in [50,100,200,500,1000,1500]:
    for dep in [None,1,2,3,4,8,9,10,12,13,15]:
        # 模型参数设置
        rf_clf = ExtraTreesClassifier(n_estimators=estimators,
                                      max_depth=dep,
                                      random_state=10,
                                      oob_score=True,
                                      bootstrap=True)

        rf_clf.fit(X_train, y_train)  # 模型拟合
        acc = accuracy_score(y_test, (rf_clf.predict(X_test)))

        if acc > n_max:
            n_max = acc
            m_estimators = estimators
            n_dep = dep

        print("运行中...")


print("n_estimators=", m_estimators)
print("max_dep=", n_dep)
print(n_max)


