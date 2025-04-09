import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

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


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris = pd.read_csv(
        './data/data_train.csv')

    X = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5', '1.55',
             '1.6', '1.65', '1.7', '1.75', '1.8', '1.85', '1.9', '1.95', '2', '2.05']]
    Y = iris['state']  # 不能用.values,会报错
    np.set_printoptions(threshold=np.inf)
    select_from_model(X, Y)  # 带特征的筛选x_data,y_data
    from sklearn.ensemble import ExtraTreesClassifier

    rf_clf = ExtraTreesClassifier(n_estimators=500, random_state=666, oob_score=True, bootstrap=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    rf_clf.fit(X_train, y_train)
    # 输出预测值
    # print(rf_clf.predict(X_test))
    # 输出预测准确率
    print("预测准确率：", accuracy_score(y_test, rf_clf.predict(X_test)))
    # print(rf_clf.oob_score_)

    nX = iris[['0.95', '1', '1.05', '1.1', '1.15', '1.2',  '1.8', '1.95', '2', '2.05']]
    nY = iris['state']
    nX_train, nX_test, ny_train, ny_test = train_test_split(nX, nY, test_size=0.25)
    rf_clf.fit(nX_train, ny_train)
    print("预测准确率：", accuracy_score(ny_test, rf_clf.predict(nX_test)))



