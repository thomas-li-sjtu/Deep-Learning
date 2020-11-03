from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pydot
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

num_features = 561  # 原始数据中特征的数目
x_train_path = './dataset/UCI HAR Dataset/train/X_train.txt'
y_train_path = './dataset/UCI HAR Dataset/train/Y_train.txt'
x_test_path = './dataset/UCI HAR Dataset/test/X_test.txt'
y_test_path = './dataset/UCI HAR Dataset/test/Y_test.txt'
columns_path = './dataset/UCI HAR Dataset/features.txt'


def arem_data_preprocess():
    pass


def txt_to_csv(x_path, y_path, columns_path, options='train'):
    """将原始的txt文件转为csv文件"""
    x_file = open(x_path, 'r')  # 读取数据与标签
    y_file = open(y_path, 'r')
    columns = open(columns_path, 'r')  # 读取属性名称

    x_lines = x_file.read().split('\n')  # 数据按行分割
    y_lines = y_file.read().split('\n')
    columns = columns.read().split('\n')

    if len(x_lines) != len(y_lines):  # 检查数据和标签是否齐全
        print("error")
        exit(1)
    with open('dataset/UCI HAR Dataset/train/{}.csv'.format(options), 'w') as csv_write:  # 写入csv
        for i in range(len(columns)):  # 写入属性名称
            if ',' in columns[i]:  # 将原有的','替换掉
                columns[i] = columns[i].replace(',', '')
            if columns[i] != '':
                if i == len(columns) - 2:
                    csv_write.write(columns[i])
                else:
                    csv_write.write(columns[i] + ',')
        csv_write.write(',labels')
        csv_write.write('\n')

        for i in range(len(x_lines)):  # 将数据与标签导入csv文件
            tmp = x_lines[i].split(' ')  # 将x的每行拆分为特征列表
            for feature in tmp:
                if feature != '':
                    csv_write.write(feature + ',')
            csv_write.write(y_lines[i] + '\n')
    print('txt changed to csv finished')


def data_to_pic(dataframe):
    """填充列表，并将每一条数据整理为24*24*1的图片形式，返回两个ndarray"""
    columns = [column for column in dataframe]
    columns.pop()  # 获得labels以外的列名
    labels = dataframe['labels']  # 获得所有labels
    data = dataframe.iloc[:, 0:561]  # 获得labels以外的数据
    for i in range(24 * 24 - 561):  # 数据扩充到24*24
        tmp = [1] * (numpy.shape(dataframe.values)[0])
        data[str(i)] = pandas.DataFrame(numpy.array(tmp))

    data_pic = []
    data = data.values
    for i in range(len(data)):  # 形成（7352,24,24）的图片
        tmp = data[i].reshape((24, 24))
        data_pic.append(tmp.tolist())

    return numpy.array(data_pic), labels.values  # (7352, 24, 24)  (7352,)


def discrete(dataframe, threshold=1023):
    """将连续数据离散化，返回一个dataframe"""
    columns = [column for column in dataframe]
    columns.pop()
    data = dataframe.iloc[:, 0:561]  # 获得labels以外的数据

    dis_data = []  # 离散后的结果
    for i in range(numpy.shape(data.values)[1]):  # 按列进行离散，离散结果为[0, 1023]
        tmp = data.iloc[:, i]
        maxi = tmp.max()
        mini = tmp.min()
        tmp_data = []
        for j in range(len(tmp.values)):  # 映射为离散数值
            tmp_data.append(int(1023 / (maxi - mini) * (float(tmp.values[j]) - mini)))
        dis_data.append(tmp_data)
    dis_data = numpy.array(dis_data)
    dis_data = dis_data.transpose()  # 转置
    dis_data = pandas.DataFrame(dis_data, index=None, columns=columns)
    dis_data['labels'] = dataframe['labels']  # 为离散后的数据添加标签
    print("discrete done")
    return dis_data


def correlation(dataframe, map='heatmap', method='pearson'):
    """查看属性之间的关联性(heatmap或colorbar可视化关联性)"""
    if method in ['pearson', 'kendall', 'spearman']:
        cor = dataframe.corr(method=method)
    else:
        raise IndexError

    fig = plt.figure(num=1, figsize=(20, 20))
    if map == 'colormap':
        plt.imshow(cor)
        plt.colorbar()
        plt.savefig("./correlation/diabetes/{}/data_correlation_heatmap_{}".format(method, method), bbox='tight')
    elif map == 'heatmap':
        seaborn.heatmap(data=cor, annot=False)
        plt.savefig("./correlation/diabetes/{}/data_correlation_heatmap_{}".format(method, method), bbox='tight')
    else:
        raise IndexError

    cor.to_csv("./correlation/diabetes/{}/cor_{}.csv".format(method, method), index=None,
               columns=[column for column in dataframe])


def reduction_pre(attribute_to_drop=None):
    """将属性和对应数据删除"""
    if attribute_to_drop is None:
        attribute_to_drop = ['Jerk']
    train_reduc = open('./dataset/UCI HAR Dataset/train/train_reduc.csv', 'w')
    with open('./dataset/UCI HAR Dataset/train/train.csv', 'r') as file:
        contents = file.read().split('\n')  # 按行划分所有内容
        columns = contents[0].split(',')  # 获得属性名
        index_to_drop = []  # 记录删除的列号
        data = [contents[i].split(',') for i in range(1, len(contents))]

        for i in range(len(columns)):  # 删除jerk，记录对应列名
            for name in attribute_to_drop:
                if name in columns[i]:
                    columns[i] = ''
                    index_to_drop.append(i)

        for i in range(len(data)):  # 删除对应列的数据
            for j in range(len(data[i])):
                if j in index_to_drop:
                    data[i][j] = ''

        for j in range(len(columns)):  # 重新写入属性名
            if columns[j] == '':
                continue
            if j == len(columns) - 1:
                train_reduc.write(columns[j] + '\n')
            else:
                train_reduc.write(columns[j] + ',')

        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] == '':
                    continue
                if j == len(data[i]) - 1:
                    train_reduc.write(data[i][j] + '\n')
                else:
                    train_reduc.write(data[i][j] + ',')
    train_reduc.close()
    print("drop attitudes finished")


def pca_reduction(data, components=20):
    """PCA降维  可以绘制曲线"""
    for components in [20, 30, 40, 50, 100]:
        pca_sk = PCA(n_components=components)
        newMat = pca_sk.fit_transform(data)  # 利用PCA进行降维，数据存在newMat中
        print(sum(pca_sk.explained_variance_ratio_))
    dataDf = pandas.DataFrame(newMat)
    dataDf.to_csv('./dataset/UCI HAR Dataset/PCA_cluster.csv')  # 数据保存在excel文件中
    return newMat


def draw_confmx(conf_mx, class_name, classi_alg, err_matrix=False):
    """绘制混淆矩阵"""
    plt.figure(figsize=(15, 15))
    plt.imshow(conf_mx, cmap=plt.cm.get_cmap('Greens'))  # 显示混淆矩阵
    indices = range(len(conf_mx))
    plt.xticks(indices, class_name, rotation=60)
    plt.yticks(indices, class_name)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    if err_matrix:
        for i in range(len(conf_mx)):  # 显示数据
            for j in range(len(conf_mx[i])):
                plt.text(i, j, round(conf_mx[i][j], 3))
        plt.savefig("./{}_errconfmx.png".format(classi_alg))
    else:
        for i in range(len(conf_mx)):  # 显示数据
            for j in range(len(conf_mx[i])):
                plt.text(i, j, conf_mx[i][j])
        plt.savefig("./{}_confmx.png".format(classi_alg))


def ran_forest(data, labels, options='train', path_name='ran_forest.pkl'):
    """随机森林分类  正确率0.9273"""

    def load_model(path_name):
        model = joblib.load(path_name)
        return model

    def save_model(model, path_name):
        joblib.dump(model, filename=path_name)

    if options == 'train':  # 训练
        # model = RandomForestClassifier()
        model = RandomForestClassifier(n_estimators=322, criterion='entropy', max_depth=50)
        model.fit(data, labels)
        save_model(model, path_name)
        print("model trained")
    elif options == 'test':  # 测试
        model = load_model(path_name)
        predict_labels = model.predict(data)
        print(model.score(data, labels))
        print(classification_report(predict_labels, labels))
        # plot_roc_curve(model, data, labels)  # 绘制roc曲线
        # plt.show()
        # plt.savefig("./ram_forest_roc_curve.png")

        conf_mx = confusion_matrix(labels, predict_labels)  # 计算混淆矩阵
        row_sums = numpy.sum(conf_mx, axis=1)  # 错误分类的百分数
        err_matrix = conf_mx / row_sums
        numpy.fill_diagonal(err_matrix, 0)
        draw_confmx(conf_mx, class_name=[1, 2, 3, 4, 5, 6], classi_alg='ran_forest')
        draw_confmx(err_matrix, class_name=[1, 2, 3, 4, 5, 6], classi_alg='ran_forest', err_matrix=True)

    else:
        raise IndexError


def decision_tree(data, labels, options='train', model_path='dctree.pkl', columns_name=None):
    """决策树分类"""
    if options == 'train':
        # dtc = DecisionTreeClassifier()
        dtc = DecisionTreeClassifier(criterion='entropy', max_depth=20, max_features=None, splitter='random')
        dtc.fit(x_train, y_train)
        joblib.dump(dtc, filename=model_path)

        y_importances = dtc.feature_importances_  # 决策树各特征权重可视化  决策树特征权重：即决策树中每个特征单独的分类能力。
        x_importances = [i for i in range(561)]
        y_pos = numpy.arange(len(x_importances))
        # 横向柱状图
        plt.figure(figsize=(90, 90))
        plt.barh(y_pos, y_importances, align='center')
        plt.yticks(y_pos, x_importances)
        plt.xlabel('Importances')
        plt.xlim(0, 1)
        plt.title('Features Importances')
        plt.savefig("./dtc_Features Importances.png")

        dot_data = StringIO()  # 决策树可视化
        tree.export_graphviz(dtc, out_file=dot_data,
                             feature_names=columns_name,
                             class_names=['1', '2', '3', '4', '5', '6'],
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("dtc.pdf")

    elif options == 'test':  # 测试
        model = joblib.load(model_path)
        predict_labels = model.predict(data)

        print(model.score(data, labels))
        print(classification_report(predict_labels, labels))

        conf_mx = confusion_matrix(labels, predict_labels)  # 计算混淆矩阵
        row_sums = numpy.sum(conf_mx, axis=1)  # 错误分类的百分数
        err_matrix = conf_mx / row_sums
        numpy.fill_diagonal(err_matrix, 0)
        draw_confmx(conf_mx, class_name=[1, 2, 3, 4, 5, 6], classi_alg='dcf')
        draw_confmx(err_matrix, class_name=[1, 2, 3, 4, 5, 6], classi_alg='dcf', err_matrix=True)
    else:
        raise IndexError


def mlp_classification(data, labels, options='train', model_path='mlp_model.pkl', hidden_size=None):
    """mlp分类  正确率0.948"""
    if hidden_size is None:
        hidden_size = (500, 200, 6)

    if options == 'train':
        kf = KFold(n_splits=5)  # 5折交叉验证
        best_clf = None
        best_score = 0
        train_scores = []
        test_scores = []

        for train_index, test_index in kf.split(data):
            # clf = MLPClassifier()
            clf = MLPClassifier(solver='sgd', activation='relu', max_iter=200, hidden_layer_sizes=hidden_size, verbose=True)
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            clf.fit(x_train, y_train)

            train_score = clf.score(x_train, y_train)  # 保存训练分数
            train_scores.append(train_score)

            test_score = clf.score(x_test, y_test)  # 保存5折的测试分数
            test_scores.append(test_score)

            if test_score > best_score:
                best_score = test_score
                best_clf = clf

        in_sample_error = [1 - score for score in train_scores]  # 打印训练中的score
        test_set_error = [1 - score for score in test_scores]
        print("in_sample_error: ")
        print(in_sample_error)
        print("test_set_error: ")
        print(test_set_error)

        if best_clf is not None:  # 保存模型
            joblib.dump(best_clf, model_path)
    elif options == 'test':  # 测试
        # 加载模型
        model = joblib.load(model_path)
        predict_labels = model.predict(data)
        print(classification_report(predict_labels, labels))

        conf_mx = confusion_matrix(labels, predict_labels)  # 计算混淆矩阵
        row_sums = numpy.sum(conf_mx, axis=1)  # 错误分类的百分数
        err_matrix = conf_mx / row_sums
        numpy.fill_diagonal(err_matrix, 0)
        draw_confmx(conf_mx, class_name=[1, 2, 3, 4, 5, 6], classi_alg='mlp')
        draw_confmx(err_matrix, class_name=[1, 2, 3, 4, 5, 6], classi_alg='mlp', err_matrix=True)
    else:
        raise IndexError


if __name__ == '__main__':
    dataframe_train = pandas.read_csv('dataset/UCI HAR Dataset/train/train.csv')
    columns = [i for i in dataframe_train]
    columns.pop()

    x_train = dataframe_train.iloc[:, 0:561].values
    y_train = dataframe_train['labels'].values

    dataframe_test = pandas.read_csv('dataset/UCI HAR Dataset/train/test.csv')
    x_test = dataframe_test.iloc[:, 0:561].values
    y_test = dataframe_test['labels'].values

    # ran_forest(x_train, y_train, options='train')
    # ran_forest(x_test, y_test, options='test')
    # decision_tree(x_train, y_train, options='train', columns_name=columns)
    # decision_tree(x_test, y_test, options='test')
    mlp_classification(x_train, y_train, options='train')
    mlp_classification(x_test, y_test, options='test')


