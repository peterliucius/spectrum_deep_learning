import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import datetime

# 筛选特征值
def select_from_model(x_data, y_data):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier

    # 使用ExtraTrees作为特征筛选的依据
    sf_model: SelectFromModel = SelectFromModel(ExtraTreesClassifier())
    sf_model.fit(x_data, y_data)
    print("建议保留的特征: ", x_data.columns[sf_model.get_support()])
    print("特征重要性：", sf_model.estimator_.feature_importances_)
    sf_model.threshold_
    sf_model.get_support()  # get_support函数来得到到底是那几列被选中了
    return sf_model.transform(x_data)  # 得到筛选的特征


iris = datasets.load_iris()
iris = pd.read_csv('./data/data_train.csv')

X = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
          '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']]
Y = iris['state']
np.set_printoptions(threshold=np.inf)
start_time = datetime.datetime.now()  # 计时开始
select_from_model(X, Y)  # 带特征的筛选x_data,y_data
end_time = datetime.datetime.now()  # 计时结束
t1 = round((end_time - start_time).microseconds / 1000)

# 模型参数设置
rf_clf = ExtraTreesClassifier(n_estimators=180,
                              max_depth=None,
                              random_state=10,
                              oob_score=True,
                              bootstrap=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.25)  # 分割测试集与训练集

rf_clf.fit(X_train, y_train)  # 模型拟合

# 输出预测值
# print(rf_clf.predict(X_test))
# 输出预测准确率
print("预测准确率1：", accuracy_score(y_test, rf_clf.predict(X_test)))
# print(rf_clf.oob_score_)

nX = iris[['1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']].values
nY = iris['state'].values
nX_train, nX_test, ny_train, ny_test = train_test_split(nX, nY, random_state=10, test_size=0.25)

start_time = datetime.datetime.now()  # 计时开始
rf_clf.fit(nX_train, ny_train)
end_time = datetime.datetime.now()  # 计时结束
t2 = round((end_time - start_time).microseconds / 1000)
print("用时：" + str(t1 + t2) + 'ms')

print("预测准确率2：", accuracy_score(ny_test, rf_clf.predict(nX_test)))

y_pred = rf_clf.predict_proba(nX_test)[:, 1]
n_rows = y_pred.shape[0]

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
    if output[k] == 1 and ny_test[k] == 1:
        tp = tp + 1
    elif output[k] == 0 and ny_test[k] == 0:
        tn = tn + 1
    elif output[k] == 1 and ny_test[k] == 0:
        fp = fp + 1
    elif output[k] == 0 and ny_test[k] == 1:
        fn = fn + 1
    else:
        print('错误分类样本的序号：', k + 1)

print("精确率：", tp/(tp + fp))
print("召回率：", tp/(tp + fn))
print("负类召回率：", tn/(tn + fp))




