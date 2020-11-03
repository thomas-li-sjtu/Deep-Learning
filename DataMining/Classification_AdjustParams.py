import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 训练集基于5折交叉验证做网格搜索找出最优参数，应用于测试集以评价算法性能
dataframe_train = pd.read_csv('dataset/UCI HAR Dataset/train/train.csv')
x_train = dataframe_train.iloc[:, 0:561].values
y_train = dataframe_train['labels'].values

dataframe_test = pd.read_csv('dataset/UCI HAR Dataset/train/test.csv')
x_test = dataframe_test.iloc[:, 0:561].values
y_test = dataframe_test['labels'].values


def adjust_ran_forest():
    # Set the parameters by cross-validation
    parameter_space = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=500, num=10)],
        "criterion": ["gini", "entropy"],
        'max_depth': [int(x) for x in np.linspace(10, 100, num=10)],
        'max_features': ['auto', 'sqrt']
    }

    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomForestClassifier(n_estimators=322)
        grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s_macro' % score, n_jobs=1)
            # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
        grid.fit(x_train, y_train)

        print_best_parameters(grid)


def adjust_decision_tree():
    parameter_space = {
        "splitter": ["best", "random"],
        "criterion": ["gini", "entropy"],
        'max_depth': [int(x) for x in np.linspace(10, 100, num=10)],
        'max_features': ['auto', 'sqrt', "log2", None]
    }

    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = DecisionTreeClassifier()
        grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s_macro' % score, n_jobs=1)
        # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
        grid.fit(x_train, y_train)

        print_best_parameters(grid)

def adjust_mlp():
    # Set the parameters by cross-validation
    parameter_space = {
        "hidden_layer_sizes": [(600, 300, 6), (300, 200, 6), (1000, 500, 100, 6)],
        "activation": ['tanh', 'relu'],
        "solver": ["sgd", "adam"],
        'learning_rate': ['constant', 'adaptive'],
        'early_stopping': [False, True]
    }

    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = MLPClassifier()
        grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s_macro' % score, n_jobs=1)
        # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
        grid.fit(x_train, y_train)

        print_best_parameters(grid)

def print_best_parameters(grid):
    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    bclf = grid.best_estimator_
    bclf.fit(x_train, y_train)
    y_true = y_test
    y_pred = bclf.predict(x_test)
    print(classification_report(y_true, y_pred))


adjust_decision_tree()
adjust_mlp()



###################### 随机森林调参结果 #######################################
"""
# Tuning hyper-parameters for precision

Best parameters set found on development set:

{'n_estimators': 322}

Grid scores on development set:

0.922 (+/-0.033) for {'n_estimators': 100}
0.922 (+/-0.032) for {'n_estimators': 144}
0.920 (+/-0.036) for {'n_estimators': 188}
0.923 (+/-0.039) for {'n_estimators': 233}
0.923 (+/-0.034) for {'n_estimators': 277}
0.923 (+/-0.040) for {'n_estimators': 322}
0.923 (+/-0.034) for {'n_estimators': 366}
0.923 (+/-0.033) for {'n_estimators': 411}
0.922 (+/-0.031) for {'n_estimators': 455}
0.923 (+/-0.031) for {'n_estimators': 500}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

              precision    recall  f1-score   support

           1       0.90      0.97      0.93       496
           2       0.89      0.92      0.90       471
           3       0.96      0.84      0.90       420
           4       0.90      0.90      0.90       491
           5       0.91      0.91      0.91       532
           6       1.00      1.00      1.00       537

    accuracy                           0.93      2947
   macro avg       0.93      0.92      0.92      2947
weighted avg       0.93      0.93      0.93      2947
"""


"""
固定322
# Tuning hyper-parameters for precision

Best parameters set found on development set:

{'criterion': 'entropy', 'max_depth': 50}

Grid scores on development set:

0.923 (+/-0.038) for {'criterion': 'gini', 'max_depth': 10}
0.926 (+/-0.036) for {'criterion': 'gini', 'max_depth': 20}
0.923 (+/-0.032) for {'criterion': 'gini', 'max_depth': 30}
0.924 (+/-0.035) for {'criterion': 'gini', 'max_depth': 40}
0.924 (+/-0.037) for {'criterion': 'gini', 'max_depth': 50}
0.924 (+/-0.035) for {'criterion': 'gini', 'max_depth': 60}
0.922 (+/-0.032) for {'criterion': 'gini', 'max_depth': 70}
0.921 (+/-0.034) for {'criterion': 'gini', 'max_depth': 80}
0.922 (+/-0.035) for {'criterion': 'gini', 'max_depth': 90}
0.921 (+/-0.036) for {'criterion': 'gini', 'max_depth': 100}
0.929 (+/-0.032) for {'criterion': 'entropy', 'max_depth': 10}
0.928 (+/-0.037) for {'criterion': 'entropy', 'max_depth': 20}
0.931 (+/-0.034) for {'criterion': 'entropy', 'max_depth': 30}
0.931 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 40}
0.933 (+/-0.032) for {'criterion': 'entropy', 'max_depth': 50}
0.931 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 60}
0.932 (+/-0.035) for {'criterion': 'entropy', 'max_depth': 70}
0.932 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 80}
0.932 (+/-0.032) for {'criterion': 'entropy', 'max_depth': 90}
0.932 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 100}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

              precision    recall  f1-score   support

           1       0.88      0.97      0.93       496
           2       0.89      0.90      0.89       471
           3       0.96      0.83      0.89       420
           4       0.93      0.90      0.91       491
           5       0.91      0.94      0.92       532
           6       1.00      1.00      1.00       537

    accuracy                           0.93      2947
   macro avg       0.93      0.92      0.93      2947
weighted avg       0.93      0.93      0.93      2947



# Tuning hyper-parameters for recall

Best parameters set found on development set:

{'criterion': 'entropy', 'max_depth': 90}

Grid scores on development set:

0.915 (+/-0.038) for {'criterion': 'gini', 'max_depth': 10}
0.918 (+/-0.036) for {'criterion': 'gini', 'max_depth': 20}
0.917 (+/-0.041) for {'criterion': 'gini', 'max_depth': 30}
0.917 (+/-0.041) for {'criterion': 'gini', 'max_depth': 40}
0.918 (+/-0.036) for {'criterion': 'gini', 'max_depth': 50}
0.917 (+/-0.041) for {'criterion': 'gini', 'max_depth': 60}
0.918 (+/-0.037) for {'criterion': 'gini', 'max_depth': 70}
0.915 (+/-0.039) for {'criterion': 'gini', 'max_depth': 80}
0.916 (+/-0.040) for {'criterion': 'gini', 'max_depth': 90}
0.917 (+/-0.038) for {'criterion': 'gini', 'max_depth': 100}
0.927 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 10}
0.927 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 20}
0.926 (+/-0.036) for {'criterion': 'entropy', 'max_depth': 30}
0.926 (+/-0.037) for {'criterion': 'entropy', 'max_depth': 40}
0.926 (+/-0.034) for {'criterion': 'entropy', 'max_depth': 50}
0.927 (+/-0.036) for {'criterion': 'entropy', 'max_depth': 60}
0.926 (+/-0.036) for {'criterion': 'entropy', 'max_depth': 70}
0.927 (+/-0.039) for {'criterion': 'entropy', 'max_depth': 80}
0.928 (+/-0.039) for {'criterion': 'entropy', 'max_depth': 90}
0.925 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 100}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

              precision    recall  f1-score   support

           1       0.89      0.98      0.93       496
           2       0.89      0.90      0.90       471
           3       0.97      0.84      0.90       420
           4       0.92      0.89      0.90       491
           5       0.90      0.93      0.91       532
           6       1.00      1.00      1.00       537

    accuracy                           0.93      2947
   macro avg       0.93      0.92      0.92      2947
weighted avg       0.93      0.93      0.93      2947


Process finished with exit code 0

"""


#################### 决策树调参结果 ##############################
"""
# Tuning hyper-parameters for precision

Best parameters set found on development set:

{'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'splitter': 'random'}

Grid scores on development set:

0.849 (+/-0.046) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'best'}
0.751 (+/-0.082) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'random'}
0.825 (+/-0.071) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'best'}
0.776 (+/-0.063) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'random'}
0.782 (+/-0.111) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'best'}
0.704 (+/-0.076) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'random'}
0.844 (+/-0.070) for {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'splitter': 'best'}
0.859 (+/-0.041) for {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'splitter': 'random'}
0.835 (+/-0.046) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'best'}
0.777 (+/-0.058) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'random'}
0.826 (+/-0.034) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'best'}
0.784 (+/-0.063) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'random'}
0.765 (+/-0.097) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'best'}
0.735 (+/-0.053) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'random'}
0.845 (+/-0.062) for {'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'splitter': 'best'}
0.829 (+/-0.064) for {'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'splitter': 'random'}
0.815 (+/-0.033) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'best'}
0.795 (+/-0.074) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'random'}
0.839 (+/-0.067) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'best'}
0.773 (+/-0.036) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'random'}
0.794 (+/-0.078) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'best'}
0.709 (+/-0.057) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'random'}
0.846 (+/-0.074) for {'criterion': 'gini', 'max_depth': 30, 'max_features': None, 'splitter': 'best'}
0.838 (+/-0.050) for {'criterion': 'gini', 'max_depth': 30, 'max_features': None, 'splitter': 'random'}
0.847 (+/-0.057) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'best'}
0.811 (+/-0.032) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'random'}
0.849 (+/-0.079) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'best'}
0.787 (+/-0.049) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'random'}
0.796 (+/-0.063) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'best'}
0.734 (+/-0.056) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.068) for {'criterion': 'gini', 'max_depth': 40, 'max_features': None, 'splitter': 'best'}
0.844 (+/-0.046) for {'criterion': 'gini', 'max_depth': 40, 'max_features': None, 'splitter': 'random'}
0.825 (+/-0.036) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'best'}
0.780 (+/-0.060) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'random'}
0.838 (+/-0.050) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'best'}
0.795 (+/-0.034) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'random'}
0.780 (+/-0.064) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'best'}
0.709 (+/-0.058) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'random'}
0.840 (+/-0.064) for {'criterion': 'gini', 'max_depth': 50, 'max_features': None, 'splitter': 'best'}
0.849 (+/-0.055) for {'criterion': 'gini', 'max_depth': 50, 'max_features': None, 'splitter': 'random'}
0.828 (+/-0.046) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'best'}
0.793 (+/-0.049) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'random'}
0.821 (+/-0.042) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'best'}
0.786 (+/-0.086) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'random'}
0.770 (+/-0.073) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'best'}
0.739 (+/-0.060) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'random'}
0.846 (+/-0.076) for {'criterion': 'gini', 'max_depth': 60, 'max_features': None, 'splitter': 'best'}
0.825 (+/-0.039) for {'criterion': 'gini', 'max_depth': 60, 'max_features': None, 'splitter': 'random'}
0.836 (+/-0.035) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'best'}
0.776 (+/-0.032) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'random'}
0.837 (+/-0.053) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'best'}
0.778 (+/-0.073) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'random'}
0.773 (+/-0.072) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'best'}
0.761 (+/-0.053) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'random'}
0.849 (+/-0.063) for {'criterion': 'gini', 'max_depth': 70, 'max_features': None, 'splitter': 'best'}
0.861 (+/-0.050) for {'criterion': 'gini', 'max_depth': 70, 'max_features': None, 'splitter': 'random'}
0.817 (+/-0.048) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'best'}
0.806 (+/-0.049) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'random'}
0.830 (+/-0.042) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'best'}
0.788 (+/-0.049) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'random'}
0.770 (+/-0.075) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'best'}
0.720 (+/-0.055) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'random'}
0.844 (+/-0.076) for {'criterion': 'gini', 'max_depth': 80, 'max_features': None, 'splitter': 'best'}
0.856 (+/-0.030) for {'criterion': 'gini', 'max_depth': 80, 'max_features': None, 'splitter': 'random'}
0.828 (+/-0.049) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'best'}
0.804 (+/-0.041) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'random'}
0.828 (+/-0.041) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'best'}
0.811 (+/-0.057) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'random'}
0.802 (+/-0.048) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'best'}
0.752 (+/-0.060) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'random'}
0.842 (+/-0.084) for {'criterion': 'gini', 'max_depth': 90, 'max_features': None, 'splitter': 'best'}
0.844 (+/-0.048) for {'criterion': 'gini', 'max_depth': 90, 'max_features': None, 'splitter': 'random'}
0.825 (+/-0.037) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'best'}
0.801 (+/-0.089) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'random'}
0.809 (+/-0.041) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'best'}
0.810 (+/-0.052) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'random'}
0.794 (+/-0.082) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'best'}
0.733 (+/-0.051) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'random'}
0.841 (+/-0.076) for {'criterion': 'gini', 'max_depth': 100, 'max_features': None, 'splitter': 'best'}
0.861 (+/-0.045) for {'criterion': 'gini', 'max_depth': 100, 'max_features': None, 'splitter': 'random'}
0.825 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'best'}
0.802 (+/-0.035) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'random'}
0.834 (+/-0.069) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'best'}
0.786 (+/-0.095) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'random'}
0.773 (+/-0.072) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'best'}
0.710 (+/-0.116) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'random'}
0.860 (+/-0.054) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'splitter': 'best'}
0.851 (+/-0.041) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'splitter': 'random'}
0.834 (+/-0.033) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'best'}
0.789 (+/-0.044) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'random'}
0.832 (+/-0.036) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'best'}
0.796 (+/-0.049) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'random'}
0.782 (+/-0.066) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'best'}
0.732 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.069) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'splitter': 'best'}
0.868 (+/-0.029) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'splitter': 'random'}
0.841 (+/-0.060) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'best'}
0.780 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'random'}
0.835 (+/-0.031) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'best'}
0.805 (+/-0.067) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'random'}
0.781 (+/-0.057) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'best'}
0.746 (+/-0.047) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'random'}
0.859 (+/-0.050) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'splitter': 'best'}
0.855 (+/-0.024) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'splitter': 'random'}
0.830 (+/-0.066) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'best'}
0.817 (+/-0.041) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'random'}
0.820 (+/-0.021) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'best'}
0.777 (+/-0.064) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'random'}
0.786 (+/-0.081) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'best'}
0.747 (+/-0.082) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'random'}
0.863 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': None, 'splitter': 'best'}
0.853 (+/-0.019) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': None, 'splitter': 'random'}
0.822 (+/-0.050) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'best'}
0.812 (+/-0.064) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'random'}
0.794 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'best'}
0.802 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'random'}
0.788 (+/-0.090) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'best'}
0.748 (+/-0.077) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'random'}
0.854 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': None, 'splitter': 'best'}
0.860 (+/-0.036) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': None, 'splitter': 'random'}
0.833 (+/-0.046) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'best'}
0.813 (+/-0.072) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'random'}
0.842 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'best'}
0.804 (+/-0.063) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'random'}
0.774 (+/-0.057) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'best'}
0.756 (+/-0.075) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'random'}
0.858 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': None, 'splitter': 'best'}
0.832 (+/-0.028) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': None, 'splitter': 'random'}
0.835 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'best'}
0.818 (+/-0.080) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'random'}
0.841 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'best'}
0.793 (+/-0.029) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'random'}
0.787 (+/-0.083) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'best'}
0.730 (+/-0.055) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'random'}
0.857 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': None, 'splitter': 'best'}
0.857 (+/-0.055) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': None, 'splitter': 'random'}
0.840 (+/-0.043) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'best'}
0.794 (+/-0.045) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'random'}
0.850 (+/-0.041) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'best'}
0.801 (+/-0.077) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'random'}
0.767 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'best'}
0.734 (+/-0.079) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'random'}
0.859 (+/-0.055) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': None, 'splitter': 'best'}
0.853 (+/-0.043) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': None, 'splitter': 'random'}
0.849 (+/-0.057) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'best'}
0.793 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'random'}
0.827 (+/-0.093) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'best'}
0.791 (+/-0.050) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'random'}
0.787 (+/-0.048) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'best'}
0.726 (+/-0.052) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.048) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': None, 'splitter': 'best'}
0.863 (+/-0.039) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': None, 'splitter': 'random'}
0.832 (+/-0.042) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'best'}
0.798 (+/-0.044) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'random'}
0.821 (+/-0.064) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'best'}
0.799 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'random'}
0.788 (+/-0.037) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'best'}
0.754 (+/-0.064) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'random'}
0.858 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': None, 'splitter': 'best'}
0.861 (+/-0.043) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': None, 'splitter': 'random'}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

              precision    recall  f1-score   support

           1       0.81      0.89      0.85       496
           2       0.81      0.74      0.77       471
           3       0.86      0.85      0.85       420
           4       0.76      0.78      0.77       491
           5       0.79      0.78      0.78       532
           6       1.00      1.00      1.00       537

    accuracy                           0.84      2947
   macro avg       0.84      0.84      0.84      2947
weighted avg       0.84      0.84      0.84      2947

# Tuning hyper-parameters for recall

Best parameters set found on development set:

{'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'splitter': 'random'}

Grid scores on development set:

0.825 (+/-0.045) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'best'}
0.780 (+/-0.118) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'random'}
0.837 (+/-0.060) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'best'}
0.782 (+/-0.076) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'random'}
0.773 (+/-0.102) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'best'}
0.698 (+/-0.075) for {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'random'}
0.850 (+/-0.069) for {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'splitter': 'best'}
0.837 (+/-0.040) for {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'splitter': 'random'}
0.841 (+/-0.031) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'best'}
0.781 (+/-0.033) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'random'}
0.791 (+/-0.056) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'best'}
0.792 (+/-0.029) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'random'}
0.797 (+/-0.081) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'best'}
0.726 (+/-0.077) for {'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'random'}
0.836 (+/-0.079) for {'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'splitter': 'best'}
0.849 (+/-0.055) for {'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'splitter': 'random'}
0.813 (+/-0.049) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'best'}
0.778 (+/-0.054) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'random'}
0.827 (+/-0.046) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'best'}
0.784 (+/-0.031) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'random'}
0.766 (+/-0.055) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'best'}
0.735 (+/-0.045) for {'criterion': 'gini', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'random'}
0.838 (+/-0.078) for {'criterion': 'gini', 'max_depth': 30, 'max_features': None, 'splitter': 'best'}
0.843 (+/-0.084) for {'criterion': 'gini', 'max_depth': 30, 'max_features': None, 'splitter': 'random'}
0.806 (+/-0.081) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'best'}
0.784 (+/-0.044) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'random'}
0.827 (+/-0.041) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'best'}
0.792 (+/-0.055) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'random'}
0.758 (+/-0.059) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'best'}
0.744 (+/-0.110) for {'criterion': 'gini', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'random'}
0.838 (+/-0.079) for {'criterion': 'gini', 'max_depth': 40, 'max_features': None, 'splitter': 'best'}
0.842 (+/-0.048) for {'criterion': 'gini', 'max_depth': 40, 'max_features': None, 'splitter': 'random'}
0.823 (+/-0.074) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'best'}
0.784 (+/-0.049) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'random'}
0.817 (+/-0.050) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'best'}
0.782 (+/-0.069) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'random'}
0.782 (+/-0.062) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'best'}
0.711 (+/-0.074) for {'criterion': 'gini', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'random'}
0.835 (+/-0.076) for {'criterion': 'gini', 'max_depth': 50, 'max_features': None, 'splitter': 'best'}
0.834 (+/-0.078) for {'criterion': 'gini', 'max_depth': 50, 'max_features': None, 'splitter': 'random'}
0.826 (+/-0.069) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'best'}
0.783 (+/-0.057) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'random'}
0.809 (+/-0.068) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'best'}
0.771 (+/-0.057) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'random'}
0.778 (+/-0.031) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'best'}
0.752 (+/-0.091) for {'criterion': 'gini', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'random'}
0.842 (+/-0.066) for {'criterion': 'gini', 'max_depth': 60, 'max_features': None, 'splitter': 'best'}
0.851 (+/-0.044) for {'criterion': 'gini', 'max_depth': 60, 'max_features': None, 'splitter': 'random'}
0.816 (+/-0.053) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'best'}
0.780 (+/-0.067) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'random'}
0.813 (+/-0.097) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'best'}
0.787 (+/-0.093) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'random'}
0.787 (+/-0.037) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'best'}
0.717 (+/-0.112) for {'criterion': 'gini', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'random'}
0.839 (+/-0.067) for {'criterion': 'gini', 'max_depth': 70, 'max_features': None, 'splitter': 'best'}
0.841 (+/-0.057) for {'criterion': 'gini', 'max_depth': 70, 'max_features': None, 'splitter': 'random'}
0.828 (+/-0.037) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'best'}
0.775 (+/-0.037) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'random'}
0.822 (+/-0.063) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'best'}
0.774 (+/-0.055) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'random'}
0.757 (+/-0.085) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'best'}
0.722 (+/-0.064) for {'criterion': 'gini', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'random'}
0.839 (+/-0.083) for {'criterion': 'gini', 'max_depth': 80, 'max_features': None, 'splitter': 'best'}
0.837 (+/-0.068) for {'criterion': 'gini', 'max_depth': 80, 'max_features': None, 'splitter': 'random'}
0.833 (+/-0.080) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'best'}
0.788 (+/-0.036) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'random'}
0.802 (+/-0.079) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'best'}
0.783 (+/-0.061) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'random'}
0.769 (+/-0.098) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'best'}
0.710 (+/-0.030) for {'criterion': 'gini', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'random'}
0.836 (+/-0.069) for {'criterion': 'gini', 'max_depth': 90, 'max_features': None, 'splitter': 'best'}
0.823 (+/-0.051) for {'criterion': 'gini', 'max_depth': 90, 'max_features': None, 'splitter': 'random'}
0.834 (+/-0.056) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'best'}
0.787 (+/-0.046) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'random'}
0.827 (+/-0.029) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'best'}
0.817 (+/-0.023) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'random'}
0.783 (+/-0.109) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'best'}
0.738 (+/-0.050) for {'criterion': 'gini', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'random'}
0.837 (+/-0.077) for {'criterion': 'gini', 'max_depth': 100, 'max_features': None, 'splitter': 'best'}
0.829 (+/-0.077) for {'criterion': 'gini', 'max_depth': 100, 'max_features': None, 'splitter': 'random'}
0.836 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'best'}
0.762 (+/-0.056) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'random'}
0.811 (+/-0.080) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'best'}
0.753 (+/-0.046) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'splitter': 'random'}
0.799 (+/-0.045) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'best'}
0.709 (+/-0.144) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'splitter': 'best'}
0.843 (+/-0.060) for {'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'splitter': 'random'}
0.809 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'best'}
0.778 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'auto', 'splitter': 'random'}
0.823 (+/-0.076) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'best'}
0.810 (+/-0.074) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt', 'splitter': 'random'}
0.783 (+/-0.073) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'best'}
0.745 (+/-0.087) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'splitter': 'random'}
0.851 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'splitter': 'best'}
0.856 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'splitter': 'random'}
0.837 (+/-0.029) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'best'}
0.790 (+/-0.062) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'auto', 'splitter': 'random'}
0.832 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'best'}
0.812 (+/-0.041) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'sqrt', 'splitter': 'random'}
0.779 (+/-0.027) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'best'}
0.729 (+/-0.063) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'log2', 'splitter': 'random'}
0.849 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'splitter': 'best'}
0.861 (+/-0.069) for {'criterion': 'entropy', 'max_depth': 30, 'max_features': None, 'splitter': 'random'}
0.833 (+/-0.068) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'best'}
0.790 (+/-0.066) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'auto', 'splitter': 'random'}
0.830 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'best'}
0.795 (+/-0.032) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'splitter': 'random'}
0.772 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'best'}
0.743 (+/-0.059) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'log2', 'splitter': 'random'}
0.856 (+/-0.058) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': None, 'splitter': 'best'}
0.852 (+/-0.043) for {'criterion': 'entropy', 'max_depth': 40, 'max_features': None, 'splitter': 'random'}
0.832 (+/-0.030) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'best'}
0.797 (+/-0.051) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'auto', 'splitter': 'random'}
0.836 (+/-0.042) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'best'}
0.791 (+/-0.048) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'sqrt', 'splitter': 'random'}
0.780 (+/-0.090) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'best'}
0.739 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.067) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': None, 'splitter': 'best'}
0.847 (+/-0.052) for {'criterion': 'entropy', 'max_depth': 50, 'max_features': None, 'splitter': 'random'}
0.845 (+/-0.040) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'best'}
0.777 (+/-0.056) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'auto', 'splitter': 'random'}
0.832 (+/-0.073) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'best'}
0.813 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'sqrt', 'splitter': 'random'}
0.771 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'best'}
0.729 (+/-0.076) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': 'log2', 'splitter': 'random'}
0.848 (+/-0.055) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': None, 'splitter': 'best'}
0.846 (+/-0.054) for {'criterion': 'entropy', 'max_depth': 60, 'max_features': None, 'splitter': 'random'}
0.823 (+/-0.072) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'best'}
0.783 (+/-0.066) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'auto', 'splitter': 'random'}
0.831 (+/-0.085) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'best'}
0.805 (+/-0.055) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'sqrt', 'splitter': 'random'}
0.795 (+/-0.089) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'best'}
0.713 (+/-0.062) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': 'log2', 'splitter': 'random'}
0.853 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': None, 'splitter': 'best'}
0.840 (+/-0.091) for {'criterion': 'entropy', 'max_depth': 70, 'max_features': None, 'splitter': 'random'}
0.835 (+/-0.062) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'best'}
0.795 (+/-0.042) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'splitter': 'random'}
0.834 (+/-0.052) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'best'}
0.801 (+/-0.061) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'sqrt', 'splitter': 'random'}
0.794 (+/-0.072) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'best'}
0.741 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'log2', 'splitter': 'random'}
0.842 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': None, 'splitter': 'best'}
0.852 (+/-0.049) for {'criterion': 'entropy', 'max_depth': 80, 'max_features': None, 'splitter': 'random'}
0.839 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'best'}
0.788 (+/-0.044) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'auto', 'splitter': 'random'}
0.835 (+/-0.038) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'best'}
0.794 (+/-0.049) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'sqrt', 'splitter': 'random'}
0.791 (+/-0.070) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'best'}
0.749 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': 'log2', 'splitter': 'random'}
0.850 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': None, 'splitter': 'best'}
0.853 (+/-0.047) for {'criterion': 'entropy', 'max_depth': 90, 'max_features': None, 'splitter': 'random'}
0.843 (+/-0.042) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'best'}
0.764 (+/-0.076) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'auto', 'splitter': 'random'}
0.826 (+/-0.076) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'best'}
0.780 (+/-0.087) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'sqrt', 'splitter': 'random'}
0.789 (+/-0.041) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'best'}
0.724 (+/-0.048) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': 'log2', 'splitter': 'random'}
0.850 (+/-0.065) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': None, 'splitter': 'best'}
0.854 (+/-0.060) for {'criterion': 'entropy', 'max_depth': 100, 'max_features': None, 'splitter': 'random'}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

              precision    recall  f1-score   support

           1       0.76      0.90      0.82       496
           2       0.77      0.74      0.76       471
           3       0.89      0.75      0.81       420
           4       0.83      0.78      0.80       491
           5       0.81      0.85      0.83       532
           6       1.00      0.99      1.00       537

    accuracy                           0.84      2947
   macro avg       0.84      0.83      0.84      2947
weighted avg       0.84      0.84      0.84      2947

"""