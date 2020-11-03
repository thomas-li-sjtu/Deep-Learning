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
data_path = ''


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
            """ dipersed：待填补的缺失变量是否离散，默认是，则用K近邻分类器，投票选出K个邻居中最多的类别进行填补；如为连
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
        return dataframe
    else:
        raise IndexError


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
    else:
        pca_sk = PCA(n_components=component)
        newMat = pca_sk.fit_transform(data)  # 利用PCA进行降维，数据存在newMat中
        variance_ratio.append([component, sum(pca_sk.explained_variance_ratio_)])
        dataframe = pandas.DataFrame(newMat, index=None)
        dataframe.to_csv('./PCA_{}.csv'.format(component), index=None)  # 数据保存在csv文件中

    print(variance_ratio)

    return dataframe


def datapreprocess(k=20):
    """数据预处理"""
    df = pandas.read_csv(raw_data_path, delimiter=',')
    df, attributes_dict = get_available_attitude(df)
    df, attributes = modify_attributes(df, attributes_dict)
    df = fill_missing(df, metric='both', neighbors=k)
    return df


if __name__ == '__main__':
    dataframe = datapreprocess()
    pca(dataframe, component=3, show_all=True)




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