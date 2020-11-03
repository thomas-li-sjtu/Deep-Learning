import pandas
import numpy
import seaborn
import pydot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import tree, neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

data = pandas.read_csv("./dataset/data.csv", encoding='utf-8')
columns = [i for i in data]

train, test = train_test_split(data, test_size=0.3)


def correlation(method='pearson'):
    """查看属性之间的关联性"""
    if method in ['pearson', 'kendall', 'spearman']:
        cor = data.corr(method=method)
    else:
        raise IndexError

    fig = plt.figure(num=1, figsize=(20, 20))
    plt.imshow(cor, cmap=plt.cm.get_cmap('Greens'))  # 显示相关性矩阵
    indices = range(len(cor))
    plt.xticks(indices, columns, rotation=60)
    plt.yticks(indices, columns)
    plt.colorbar()
    cor = cor.values
    for i in range(len(cor)):  # 显示数据
        for j in range(len(cor[i])):
            plt.text(i-0.3, j, round(cor[i][j], 3))
    plt.savefig("./data_correlation_{}".format(method), bbox='tight')

    # seaborn.set(style='whitegrid', context='notebook')  # style控制默认样式,context控制着默认的画幅大小
    # columns.pop(0)
    # seaborn.pairplot(data[columns], size=2.5)
    # plt.tight_layout()
    # plt.savefig('./pairplot.png', dpi=300)
    # plt.show()


def ran_forest(data, options="train", model_path='forest.pkl'):
    """随机森林回归"""
    if options == "train":  # 训练
        x = data.drop(columns=["id", "price"])
        y = data["price"]
        # model = RandomForestRegressor()
        model = RandomForestRegressor(n_estimators=230, max_depth=70)
        model.fit(x.values, y.values)
        joblib.dump(model, filename=model_path)  # 保存模型
    elif options == "test":  # 测试
        x = data.drop(columns=["id", "price"])
        y = data["price"]
        model = joblib.load(model_path)
        y_pred = model.predict(x)

        evaluate(model, y, y_pred)  # 评估模型
    else:
        raise IndexError


def lightGBM(train, test):
    train_x = train.drop(columns=["id", "price"])
    train_y = train["price"]
    test_x = test.drop(columns=["id", "price"])
    test_y = test["price"]
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y)

    params = {
        'task': 'train',
        'boosting_type': 'rf',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'rmse'},  # 评估函数
        'num_leaves': 78,  # 叶子节点数
        'num_iterations': 200,
        'max_depth': 8,
        'learning_rate': 0.0001,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=lgb_eval, early_stopping_rounds=5)
    pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)

    print("r2：", r2_score(test_y, pred))
    print("均方根误差：", numpy.sqrt(mean_squared_error(test_y, pred)))
    print("解释方差：", explained_variance_score(test_y, pred))
    print("平均绝对误差：", mean_absolute_error(test_y, pred))
    print("均方误差：", mean_squared_error(test_y, pred))


def decison_tree(data, options="train", model_path='tree.pkl'):
    if options == "train":  # 训练
        x = data.drop(columns=["id", "price"])
        y = data["price"]
        # model = tree.DecisionTreeRegressor()
        model = tree.DecisionTreeRegressor(criterion='friedman_mse', max_depth=9, max_features='auto', splitter='best')
        model.fit(x.values, y.values)
        joblib.dump(model, filename=model_path)  # 保存模型

    elif options == "test":  # 测试
        x = data.drop(columns=["id", "price"])
        y = data["price"]
        model = joblib.load(model_path)
        y_pred = model.predict(x)

        evaluate(model, y, y_pred)  # 评估模型
    else:
        raise IndexError


def evaluate(model, y, y_pred):
    """评估方式"""
    print("r2：", r2_score(y, y_pred))
    print("均方根误差：", numpy.sqrt(mean_squared_error(y, y_pred)))
    print("解释方差：", explained_variance_score(y, y_pred))
    print("平均绝对误差：", mean_absolute_error(y, y_pred))
    print("均方误差：", mean_squared_error(y, y_pred))

    residuals(model)


def residuals(model):
    """绘制残差图"""
    y_train_pred = model.predict(train.drop(columns=["id", "price"]))
    y_test_pred = model.predict(test.drop(columns=["id", "price"]))

    plt.scatter(y_train_pred, y_train_pred-train["price"], c='lightgreen', marker='o', label='Training data', s=2)
    plt.scatter(y_test_pred, y_test_pred-test["price"], c='blue', marker='s', label='Test data', s=2)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=1.2, lw=2, color='red')
    plt.xlim([0, 1.2])
    plt.tight_layout()
    plt.savefig('./residuals.png', dpi=300)


# for i in range(10):
#     lightGBM(train, test)
#     train, test = train_test_split(data, test_size=0.3)
for i in range(30):
    ran_forest(train)
    ran_forest(test, options="test")
    train, test = train_test_split(data, test_size=0.3)
# for i in range(20):
#     decison_tree(train)
#     decison_tree(test, options="test")
#     train, test = train_test_split(data, test_size=0.3)


