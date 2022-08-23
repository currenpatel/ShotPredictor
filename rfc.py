#https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters
import numpy as np 
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from featureSelection import *
from eval import *
import time

start_time = time.time()

X_train, X_test, y_train, y_test, df = loadData(treeFeatureReduction)

#random_sample = X_test.take(np.random.permutation(len(X_test))[:3])

def getBestRandomForest(X_test, y_test, X_train, y_train, start_time, start = 100, stop = 2000, num = 10):
    #random forest
    # best hyperparameters

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = start, stop = stop, num = num)]
    bootstrap = [True, False]

    grid = {'n_estimators': n_estimators}

    #print(*random_grid, sep='\n')
    rfc = RandomForestClassifier(min_samples_split = 2, max_features = 'sqrt', max_depth = None, n_jobs = -1)
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)
    best = GridSearchCV(estimator = rfc, param_grid = grid, cv = skf, verbose = 2, n_jobs = -1, return_train_score = True)
    best.fit(X_train, y_train)
    print("\nDONE\n")
    print("Best Params: ", best.best_params_)
    [print(key, value) for key, value in best.cv_results_.items()]
    plot_grid_1("rfc_n_estimators" , best.cv_results_, n_estimators, 'N estimators')
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred = best.predict(X_test)
    writeReport(best.cv_results_ ,X_test, X_train, y_test, y_pred , best, "rfc", (time.time() - start_time))

#find bagging hyperparameters
#50 150
def getBestBagging(X_test, y_test, X_train, y_train, start_time, start = 10, stop = 100, num = 4):
    n_estimators = [int(x) for x in np.linspace(start = start, stop = stop, num = num)]
    #first test
    max_features = np.linspace(0.1, 0.3, num=5) #(0.025, 0.2, num=3)
    max_samples = np.linspace(0.1, 0.3, num=5) #(0.025, 0.1, num=3)

    #second test
    #max_features = np.array([1.0])
    #max_samples = np.linspace(0.01, 0.1, num=3)

    #max features only
    feature_grid = {'n_estimators': n_estimators,
            'max_features': max_features}

    #max samples only
    sample_grid = {'n_estimators': n_estimators,
            'max_samples': max_samples}

    #all
    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_samples': max_samples}


    rfc = RandomForestClassifier(min_samples_split = 2, max_features = 'sqrt', max_depth = None, n_jobs = -1)
    #bagging = BaggingClassifier(rfc, n_estimators = 1000, max_features = 5)
    bagging = BaggingClassifier(rfc, random_state = 42)
    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 42)
    best = GridSearchCV(estimator = bagging, param_grid = feature_grid, cv = skf, verbose=2, n_jobs = -1, return_train_score = True)
    best.fit(X_train, y_train)
    print("Best parameters: ", best.best_params_)
    #print("Grid Search Results:\n")
    #[print(key, value) for key, value in best.cv_results_.items()]
    y_pred = best.predict(X_test)
    #plot_grid_search("bagging_max_features" , best.cv_results_, n_estimators, max_features, 'N estimators', "Max Features")
    #plot_grid_search("bagging_max_samples" , best.cv_results_, n_estimators, max_samples, 'N estimators', "Max Samples")
    writeReport(best.cv_results_ ,X_test, X_train, y_test, y_pred , best, "bagging_2", (time.time() - start_time))
    return best, best.cv_results_
 


def defaultRandomForest(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators = 1200 ,min_samples_split = 2, max_features = 'sqrt', max_depth = None, n_jobs = -1)
    rfc.fit(X_train, y_train)
    return rfc

def defaultBagging(X_train, y_train):
    rfc = RandomForestClassifier(min_samples_split = 2, max_features = 'sqrt', max_depth = None)
    rf_best = BaggingClassifier(rfc, n_estimators = 300, max_features = 0.2, max_samples = 0.2, n_jobs = -1)
    rf_best.fit(X_train, y_train)
    return rf_best
    


def featureImportanceGraph(model):
    feature_imp = pd.Series(model.feature_importances_, index = list(X.columns)).sort_values(ascending=False)
    sns.barplot(x = feature_imp, y = feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    #plt.show()

getBestBagging(X_test, y_test, X_train, y_train, start_time)#1200
#getBestRandomForest(X_test, y_test, X_train, y_train, start_time)

'''
best = defaultBagging(X_train, y_train)
y_pred = best.predict(X_test)

print(classificationReport(y_test, y_pred))
print(accuracyReport(y_test, y_pred))
print(confusionReport(y_test, y_pred))
'''