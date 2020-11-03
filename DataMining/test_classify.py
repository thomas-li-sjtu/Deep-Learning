from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot




import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

# 1.roc曲线，混淆矩阵，热图，可视化决策树

num_features = 561  # 原始数据中特征的数目
x_train_path = './dataset/UCI HAR Dataset/train/X_train.txt'
y_train_path = './dataset/UCI HAR Dataset/train/Y_train.txt'
x_test_path = './dataset/UCI HAR Dataset/test/X_test.txt'
y_test_path = './dataset/UCI HAR Dataset/test/Y_test.txt'
columns_path = './dataset/UCI HAR Dataset/features.txt'












dataframe_train = pandas.read_csv('dataset/UCI HAR Dataset/train/train.csv')
tmp = [i for i in dataframe_train]
tmp.pop()
x_train = dataframe_train.iloc[:, 0:561].values
y_train = dataframe_train['labels'].values

dataframe_test = pandas.read_csv('dataset/UCI HAR Dataset/train/test.csv')
x_test = dataframe_test.iloc[:, 0:561].values
y_test = dataframe_test['labels'].values


dot_data = StringIO()
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=tmp,
                        class_names=['1','2','3','4','5','6'],
                        filled=True, rounded=True,
                        special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")