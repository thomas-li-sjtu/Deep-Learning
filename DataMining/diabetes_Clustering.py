from sklearn import metrics
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
import sklearn.cluster as cluster
from sklearn.preprocessing import normalize

import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas
import numpy
from numpy import *


raw_data_path = "./dataset/diabetes/diabetic_data.csv"


def get_available_attitude(dataframe):
    """删除数据过少的属性，删除只有一个取值的属性
        返回属性字典和新的dataframe
    """
    data_mat = dataframe.values  # ndarray (101766, 50)
    columns = [column for column in dataframe]

    index_to_drop = []  # 要删除的属性下标
    for i in range(2, numpy.shape(data_mat)[1]):  # 查看有未知项的属性，若未知项太多，则加入index_to_drop
        tmp = data_mat[:, i].copy()
        set_states = set(tmp)  # 计数"?"
        if '?' in set_states:  # 统计当前样本中的?
            count = 0
            for attribute in tmp:
                if attribute == '?':
                    count += 1
            if count > numpy.shape(data_mat)[0] / 10:
                index_to_drop.append([i, columns[i]])
            print(columns[i])
        if len(list(set_states)) == 1:  # 如果属性取值只有一个，也需要删除
            index_to_drop.append([i, columns[i]])
    index_to_drop.append([0, columns[0]])
    index_to_drop.append([1, columns[1]])  # 前两个id属性也要删除

    for i in range(len(index_to_drop)):
        del dataframe[index_to_drop[i][1]]  # 删除列

    data_mat = dataframe.values
    attributes_dict = {}  # 属性的取值字典
    for i in range(0, numpy.shape(data_mat)[1]):  # 获得各个属性的所有可能取值
        tmp = data_mat[:, i].copy()
        set_states = set(tmp)
        set_states = sorted(set_states)
        attributes_dict[[column for column in dataframe][i]] = set_states

    return dataframe, attributes_dict


def modify_attributes(dataframe, att_dict={}):
    """将所有属性转为数字类型，返回属性list与修改后的dataframe"""
    attributes = []
    keys = list(att_dict.keys())
    values = list(att_dict.values())
    for i in range(len(keys)):  # 将所有属性的取值映射为数值
        tmp = [keys[i]]
        if isinstance(values[i][0], (int, float)):  # 属性取值为数值型
            for j in range(len(values[i])):
                tmp.append([values[i][j], values[i][j]])
        else:  # 属性取值不为数值型
            for j in range(len(values[i])):
                tmp.append([j, values[i][j]])
        attributes.append(tmp)

    with open('./dataset/diabetes/attributes.txt', 'w') as file:  # 将映射结果保存到txt
        for attribute in attributes:
            file.write(attribute[0] + '\n')
            for i in range(1, len(attribute)):
                file.write('\t')
                file.write(str(attribute[i][0]) + ' ' + str(attribute[i][1]))
                file.write('\n')

    for i in range(numpy.shape(dataframe.values)[1]):  # 将dataframe中相应内容替换掉
        tmp = dataframe[attributes[i][0]].tolist()
        for j in range(len(tmp)):  # 按列替换
            for k in range(1, len(attributes[i])):
                if tmp[j] == '?':  # 将?替换为NAN
                    tmp[j] = 'NAN'
                elif tmp[j] == attributes[i][k][1]:
                    tmp[j] = attributes[i][k][0]
        dataframe[attributes[i][0]] = pandas.Series(tmp)  # 替换

    return dataframe, attributes


def fill_missing(dataframe, metric="knn", neighbors=20):
    """填补缺失
        返回填补后的dataframe
    """
    data = dataframe.values
    columns = [column for column in dataframe]
    if metric == 'simple':  # 采用众数来填补缺失值
        imputer = SimpleImputer(missing_values='NAN', strategy='most_frequent', verbose=1)
        data = imputer.fit_transform(data)
        dataframe = pandas.DataFrame(data=data, index=None, columns=columns)
        return dataframe
    elif metric == 'knn':
        for i in range(len(data)):  # 将'NAN'转为numpy.nan，并将字符串转为float
            for j in range(len(data[i])):
                if data[i][j] == 'NAN':
                    data[i][j] = numpy.nan
                else:
                    data[i][j] = float(data[i][j])

        imputer = KNNImputer(n_neighbors=neighbors, weights='distance', metric='nan_euclidean')  # 用knn来填补缺失值
        data = imputer.fit_transform(data)
        data = pandas.DataFrame(data=data, index=None, columns=columns)
        return data
    elif metric == 'both':
        attributes = []
        for i in range(0, numpy.shape(data)[1]):  # 获得各个属性的所有可能取值
            tmp = data[:, i].copy()
            set_states = set(tmp)
            attributes.append([columns[i], set_states])

        col_tofill = []
        for i in range(len(data)):  # 取值有'NAN'的属性
            for j in range(len(data[i])):
                if data[i][j] == 'NAN':
                    col_tofill.append(columns[j])
                else:
                    data[i][j] = float(data[i][j])

        col_tofill = sorted(list(set(col_tofill)))
        col_simple = []  # ['race']
        col_knn = []  # ['diag_1', 'diag_2', 'diag_3']
        for i in range(len(attributes)):  # 得到col_simple与col_rnn
            if attributes[i][0] in col_tofill:
                if len(attributes[i][1]) < 10:
                    col_simple.append(attributes[i][0])
                else:
                    col_knn.append(attributes[i][0])

        # simpleimputer
        dataframe = pandas.DataFrame(data=data, index=None, columns=columns)
        for i in range(len(col_simple)):
            tmp = float(dataframe[col_simple[i]].mode())
            dataframe[col_simple[i]].replace('NAN', tmp, inplace=True)

        # knnimputer
        data = dataframe.values
        index_tofill = []  # 含有'NAN'的行索引
        for i in range(len(data)):  # 获得含有'NAN'的行索引
            for j in range(len(data[i])):
                if data[i][j] == 'NAN':
                    index_tofill.append(i)
        index_tofill = sorted(list(set(index_tofill)))

        x_train = []
        y_train1 = []
        y_train2 = []
        y_train3 = []
        test = []
        for i in range(len(data)):  # 划分knn的训练数据等
            tmp = []
            for j in range(len(data[i])):
                if columns[j] not in col_knn:  # 不为nan列，则作为数据
                    tmp.append(data[i][j])
                elif i not in index_tofill:  # 为nan列，则作为标签
                    if columns[j] == col_knn[0]:  # 有问题
                        y_train1.append(data[i][j])
                    elif columns[j] == col_knn[1]:
                        y_train2.append(data[i][j])
                    elif columns[j] == col_knn[2]:
                        y_train3.append(data[i][j])
                else:
                    continue
            if i not in index_tofill:
                x_train.append(tmp)
            else:
                test.append(tmp)

        def knn_missing_filled(x_train, y_train, test, k=3, dispersed=True):
            """dipersed：待填补的缺失变量是否离散，默认是，则用K近邻分类器，投票选出K个邻居中最多的类别进行填补；如为连
            续变量，则用K近邻回归器，拿K个邻居中该变量的平均值填补。"""
            if dispersed:
                clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
            else:
                clf = KNeighborsRegressor(n_neighbors=k, weights="distance")

            clf.fit(x_train, y_train)
            return clf.predict(test)

        predict1 = knn_missing_filled(x_train=x_train, y_train=y_train1, test=test, k=neighbors)
        for i in range(len(index_tofill)):
            data[index_tofill[i]][columns.index(col_knn[0])] = predict1[i]
        predict2 = knn_missing_filled(x_train=x_train, y_train=y_train2, test=test, k=neighbors)
        for i in range(len(index_tofill)):
            data[index_tofill[i]][columns.index(col_knn[1])] = predict2[i]
        predict3 = knn_missing_filled(x_train=x_train, y_train=y_train3, test=test, k=neighbors)
        for i in range(len(index_tofill)):
            data[index_tofill[i]][columns.index(col_knn[2])] = predict3[i]

        dataframe = pandas.DataFrame(data=data, index=None, columns=columns)
        dataframe = dataframe.apply(lambda x: x.astype(float))
        dataframe.info()
        dataframe.to_csv('./both.csv', index=None)
        return dataframe
    else:
        raise IndexError


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


def pca(dataframe, component=3, show_all=False):
    """pca降维"""
    data = dataframe.values
    variance_ratio = []
    if show_all:  # 生成多个pca降维结果
        for components in [7, 6, 5, 4, 3]:
            pca_sk = PCA(n_components=components)
            newMat = pca_sk.fit_transform(data)  # 利用PCA进行降维，数据存在newMat中
            variance_ratio.append([components, sum(pca_sk.explained_variance_ratio_)])
            dataframe = pandas.DataFrame(newMat, index=None)
            dataframe.to_csv('./PCA_{}.csv'.format(components), index=None)  # 数据保存在csv文件中
        print(variance_ratio)
    else:
        pca_sk = PCA(n_components=component)
        newMat = pca_sk.fit_transform(data)  # 利用PCA进行降维，数据存在newMat中
        variance_ratio.append([component, sum(pca_sk.explained_variance_ratio_)])
        dataframe = pandas.DataFrame(newMat, index=None)
        dataframe.to_csv('./PCA_{}.csv'.format(component), index=None)  # 数据保存在csv文件中

    return dataframe


def datapreprocess(k=20):
    """数据预处理"""
    df = pandas.read_csv(raw_data_path, delimiter=',')
    df, attributes_dict = get_available_attitude(df)
    df, attributes = modify_attributes(df, attributes_dict)
    df = fill_missing(df, metric='both', neighbors=k)
    return df


def clusering(dataframe, method='BIRCH'):
    """
        'num_medications', 'time_in_hospital', 'number_diagnoses'  聚类属性
        'A1Cresult' / 'max_glu_serum'  聚类后比较对象
    """
    columns_to_group = ['num_medications', 'time_in_hospital', 'number_diagnoses']
    target1 = 'A1Cresult'
    target2 = 'max_glu_serum'

    tmpdataframe1 = dataframe[
        ['num_medications', 'time_in_hospital', 'number_diagnoses', 'A1Cresult']]
    tmpdata1 = tmpdataframe1.values
    dataframe1 = []
    for i in range(len(tmpdata1)):  # 去除A1Cresul属性值为None的数据
        if tmpdata1[i][3] != 2:
            dataframe1.append(tmpdata1[i])
    dataframe1 = pandas.DataFrame(numpy.array(dataframe1), index=None,
                                  columns=['num_medications', 'time_in_hospital', 'number_diagnoses',
                                           'A1Cresult'])
    seaborn.pairplot(dataframe1, hue=target1, height=3, diag_kind='kde', markers=["o", "s", "D"])  # 绘制散点矩阵
    plt.savefig('{}_clustertype{}_pairplot.png'.format(method, target1), bbox='tight')

    tmpdataframe2 = dataframe[
        ['num_medications', 'time_in_hospital', 'number_diagnoses', 'max_glu_serum']]
    tmpdata2 = tmpdataframe2.values
    dataframe2 = []
    for i in range(len(tmpdata2)):  # 去除max_glu_serum属性值为None的数据
        if tmpdata2[i][3] != 2:
            dataframe2.append(tmpdata2[i])
    dataframe2 = pandas.DataFrame(numpy.array(dataframe2), index=None,
                                  columns=['num_medications', 'time_in_hospital', 'number_diagnoses',
                                           'max_glu_serum'])
    seaborn.pairplot(dataframe2, hue=target2, height=3, diag_kind='kde', markers=["o", "s", "D"])  # 绘制散点矩阵
    plt.savefig('{}_clustertype{}_pairplot.png'.format(method, target2), bbox='tight')


    # 以不同的算法聚类
    if method == 'Agglomerative':
        data1 = dataframe1[columns_to_group].values  # 聚类'A1Cresult'
        data1 = normalize(data1, axis=0, norm='l2')
        agg1 = cluster.AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='l2')
        labels1 = agg1.fit_predict(data1)

        data2 = dataframe2[columns_to_group].values  # 聚类'max_glu_serum'
        data2 = normalize(data2, axis=0, norm='max')
        agg2 = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')
        labels2 = agg2.fit_predict(data2)


    elif method == 'BIRCH':
        data1 = dataframe1[columns_to_group].values  # 聚类
        data1 = normalize(data1, axis=0, norm='max')
        birch1 = cluster.Birch(threshold=0.01, n_clusters=3, branching_factor=50)
        labels1 = birch1.fit_predict(data1)

        data2 = dataframe2[columns_to_group].values  # 聚类
        data2 = normalize(data2, axis=0, norm='max')
        birch2 = cluster.Birch(threshold=0.03, n_clusters=3, branching_factor=50)
        labels2 = birch2.fit_predict(data2)

    else:
        raise IndexError

    fig1 = plt.figure(figsize=(15, 15))  # 绘制三维散点图
    x = data1[:, 0]
    y = data1[:, 1]
    z = data1[:, 2]
    ax = Axes3D(fig1)
    ax.scatter(x, y, z, s=5, c=labels1, marker='d')
    plt.savefig('{}_clustertype{}_result.png'.format(method, target1), bbox='tight')

    fig2 = plt.figure(figsize=(15, 15))  # 绘制三维散点图
    x = data2[:, 0]
    y = data2[:, 1]
    z = data2[:, 2]
    ax = Axes3D(fig2)
    ax.scatter(x, y, z, s=5, c=labels2, marker='d')
    plt.savefig('{}_clustertype{}_result.png'.format(method, target2), bbox='tight')

    true_labels1 = dataframe1[target1].values.tolist()  # 查看各聚类评价参数1
    true_labels1 = [2 if i == 3 else i for i in true_labels1]
    pred_labels1 = [int(i) for i in labels1]
    rand_index = metrics.adjusted_rand_score(true_labels1, pred_labels1)
    mutual_infor = metrics.adjusted_mutual_info_score(true_labels1, pred_labels1)
    silhouette_coefficient = metrics.silhouette_score(data1, labels1, metric='euclidean')
    ch_index = metrics.calinski_harabaz_score(data1, labels1)
    fmi = metrics.fowlkes_mallows_score(true_labels1, pred_labels1)
    print('pred_labels1:')
    print('兰德系数：', rand_index)  # 值越大越好
    print('互信息：', mutual_infor)  # 值越大越好
    print('轮廓系数: ', silhouette_coefficient)  # 分数越高越好。取值为-1~1
    print('Calinski-Harabasz分数：', ch_index)  # 类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
    print('Fowlkes-Mallows分数：', fmi)

    true_labels2 = dataframe2[target2]  # 评估聚类效果2
    pred_labels2 = pandas.DataFrame(labels2, index=None, columns=['pred'])
    true_labels2 = dataframe2[target2].values.tolist()  # 查看各聚类评价参数2
    true_labels2 = [2 if i == 3 else i for i in true_labels2]
    pred_labels2 = [int(i) for i in labels2]
    rand_index = metrics.adjusted_rand_score(true_labels2, pred_labels2)
    mutual_infor = metrics.adjusted_mutual_info_score(true_labels2, pred_labels2)
    silhouette_coefficient = metrics.silhouette_score(data2, labels2, metric='euclidean')
    ch_index = metrics.calinski_harabaz_score(data2, labels2)
    fmi = metrics.fowlkes_mallows_score(true_labels2, pred_labels2)
    print('pred_labels2:')
    print('兰德系数：', rand_index)  # 值越大越好
    print('互信息：', mutual_infor)  # 值越大越好
    print('轮廓系数: ', silhouette_coefficient)  # 分数越高越好。取值为-1~1
    print('Calinski-Harabasz分数：', ch_index)  # 类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
    print('Fowlkes-Mallows分数：', fmi)


if __name__ == '__main__':
    dataframe = datapreprocess()
    correlation(dataframe)
    pca(dataframe, show_all=True)
    clusering(dataframe, method='Agglomerative')
    clusering(dataframe, method='BIRCH')


'''
def autoencoder_initial_model(input_dimension=43, hidden_layers=None, activation_function="relu", dropout=0.0):
    """模型结构设置"""
    if hidden_layers is None:
        hidden_layers = [50, 10]

    input = Input(shape=(input_dimension,))  # 输入层，维度input_dimension

    for i in range(0, len(hidden_layers)):  # encode层
        if i == 0:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(input)
        else:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)

    encoded = Dropout(dropout)(encoded)  # 层之间dropout

    for i in range(len(hidden_layers) - 1, -1, -1):  # decode层
        if i == len(hidden_layers) - 1:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)
        else:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(decoded)

    decoded = Dropout(0.2)(decoded)  # 层之间dropout

    if len(hidden_layers) == 1:  # 输出层
        decoded = Dense(input_dimension, activation="sigmoid")(encoded)
    else:
        decoded = Dense(input_dimension, activation="sigmoid")(decoded)

    autoencoder = Model(outputs=decoded, inputs=input)
    autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta")

    return autoencoder


def train_autoencoder(inputs=None, input_dimension=43, phase="training", hidden_layers=None,
                      activation_function="relu", dropout=0.0, testing_filename="", batch_size=1):
    """训练autoencoder，实现数据降维"""
    if hidden_layers is None:
        hidden_layers = [50, 10]

    def check_directory(filename, root="models"):
        """建立目录"""
        if not os.path.isdir("./{}/{}".format(root, filename)):
            os.mkdir("./{}/{}".format(root, filename))

    if phase == "training":
        numpy.random.seed(666)

        autoencoder = autoencoder_initial_model(input_dimension, hidden_layers, activation_function, dropout)  # 模型初始化
        autoencoder.fit(x=inputs, y=inputs, batch_size=batch_size, steps_per_epoch=None, shuffle=False,
                        epochs=100, verbose=1)  # 训练

        # check_directory('train_autoencoder', "models")
        autoencoder.save("./models/{}.hdf5".format('train'), overwrite=True)
        print("\nModel training Finished.")
    else:
        raise IndexError
'''