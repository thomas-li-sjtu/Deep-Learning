import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree


data = pd.read_csv("./dataset/data.csv")
train, test = train_test_split(data, test_size=0.3)

x_train = train.drop(columns=["id", "price"]).values
y_train = train["price"].values

x_test = test.drop(columns=["id", "price"]).values
y_test = test["price"].values


def adjust_ran_forest():
    parameter_space = {
        "n_estimators": [int(x) for x in np.linspace(start=50, stop=500, num=100)],
        'max_depth': [int(x) for x in np.linspace(5, 100, num=20)],
        'max_features': ['auto', 'sqrt']
    }

    clf = RandomForestRegressor()
    grid = GridSearchCV(clf, parameter_space, cv=3, scoring='r2', n_jobs=1)
    grid.fit(x_train, y_train)

    print_best_parameters(grid)


def adjust_tree():
    # Set the parameters by cross-validation
    parameter_space = {
        "splitter": ["best", "random"],
        "criterion": ["friedman_mse", "mae", "mse"],
        'max_depth': [int(x) for x in np.linspace(2, 50, num=20)],
        'max_features': ['auto', 'sqrt', "log2", None]
    }

    clf = tree.DecisionTreeRegressor()
    grid = GridSearchCV(clf, parameter_space, scoring='r2', n_jobs=1)
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


adjust_ran_forest()
adjust_tree()



