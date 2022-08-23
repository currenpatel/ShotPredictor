#https://machinelearningmastery.com/feature-selection-machine-learning-python/
import numpy as np 
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from featureSelection import *

def loadData(feature_selection_func, n = None):
    df = pd.read_csv("data/clean.csv")

    categories = df.select_dtypes([object]).columns

    #one hot
    for var in categories:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], 1)
        df = df.drop(var, 1)

    X = df.drop('shot_made_flag', 1)

    y = np.array(df['shot_made_flag']).astype(int)
    if n == None:
        X = feature_selection_func(X, y)
    else:
        X = feature_selection_func(X, y, n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("Loaded data\n")
    return X_train, X_test, y_train, y_test, df

def plot_grid_1(modelType ,cv_results, grid_param_1,name_param_1):
    # Get Test test_scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']

    # Plot Grid search test_scores
    _, ax = plt.subplots(1,1)

    ax.plot(grid_param_1, scores_mean, '-o')

    ax.set_title(modelType + " test_scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig("graphs/" + modelType + "_cv" + ".png", bbox_inches='tight')

#cv graph
def plot_grid_search(modelType ,cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test test_scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search test_scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title(modelType + " test_scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig("graphs/" + modelType + "_cv" + ".png", bbox_inches='tight')

'''
    tp fp
    fn tn
'''
def confusionReport(y_test, y_pred):
    report = ("-" * 50) + "\n"
    report += "CONFUSION MATRIX:\n\n"
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrixToString = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in matrix)+'\n'
    report += matrixToString + ("-" * 50) + "\n"
    return report

'''
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F-Score = (2 * Recall * Precision) / (Recall + Precision)
'''
def classificationReport(y_test, y_pred):
    report = ("-" * 50) + "\n"
    report += "CLASSIFICATION REPORT:\n\n"
    report += metrics.classification_report(y_test, y_pred, target_names=['0', '1'])
    report +=("-" * 50) + "\n"
    return report

def accuracyReport(y_test, y_pred):
    report = ("-" * 50) + "\n"
    report += "ACCURACY REPORT:\n\n"
    yPredCopy = np.copy(y_pred)
    report += "Accuracy: " +  str(accuracy_score(y_test, yPredCopy)) + "\n"
    yPredCopy = np.zeros(yPredCopy.shape)
    report += "Accuracy all miss: " +  str(accuracy_score(y_test, yPredCopy)) + "\n"
    yPredCopy = np.ones(yPredCopy.shape)
    report += "Accuracy all make: " + str(accuracy_score(y_test, yPredCopy)) + "\n"
    yPredCopy = np.random.choice([0, 1], y_test.shape[0], p = [0.447, 1 - 0.447]) #based on fg%
    report += "Accuracy weighted random based on fg%: " + str(accuracy_score(y_test, yPredCopy)) + "\n"
    report += ("-" * 50) + "\n"
    return report

#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def rocReport(X_test, y_test, model, modelName):
    # generate a no skill prediction (majority class)
    ns_probs = np.zeros(y_test.shape[0])
    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate test_scores
    ns_auc = metrics.roc_auc_score(y_test, ns_probs)
    lr_auc = metrics.roc_auc_score(y_test, lr_probs)
    # summarize test_scores
    report = ("-" * 50) + "\n"
    report += "ROC REPORT:\n\n"
    report += ('No Skill: ROC AUC=%.3f\n' % (ns_auc))
    report += ('Model: ROC AUC=%.3f' % (lr_auc))
    report += ("\n")
    report += ("-" * 50) + "\n"
    # calculate roc curves
    ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    figure = plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(modelName + ' ROC Curve')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig("graphs/" + modelName + "_roc" + ".png", bbox_inches='tight')
    plt.show()
    return report

def writeReport(cv_results ,X_test, X_train, y_test, y_pred ,model, modelName, runtime):
    params = cv_results['params']
    test_scores = cv_results['mean_test_score']
    train_scores = cv_results['mean_train_score']
    test_stds = cv_results['std_test_score']
    train_stds = cv_results['std_train_score']
    f = open("reports/{}_report.txt".format(modelName),"w+")
    f.truncate(0)
    f.write("START OF CV REPORT:\n")
    for i in range(0, test_scores.shape[0]):
        f.write("Iteration {}:\n".format(i))
        f.write("           Params: {}\n".format(params[i]))
        f.write("           Train Scores: {} (+/- {})\n".format(train_scores[i], train_stds[i] * 2))
        f.write("           Test Scores: {} (+/- {})\n".format(test_scores[i], test_stds[i] * 2))
    f.write(("-" * 50) + "\n")
    f.write(("Best parameters:\n" + str(model.best_params_) + "\n"))
    f.write(("-" * 50) + "\n")
    f.write(classificationReport(y_test, y_pred))
    f.write(accuracyReport(y_test, y_pred))
    f.write(confusionReport(y_test, y_pred))
    f.write(rocReport(X_test, y_test, model, modelName))
    f.write("Runtime: " + str(runtime) + " s")
    f.close()
'''
X_train, X_test, y_train, y_test, df = loadData(recursiveFeatureReduction)

best = RandomForestClassifier(n_estimators = 250, min_samples_split = 2, max_features = 'sqrt', max_depth = None)
#best = BaggingClassifier(best, n_estimators = 100, max_features = 5, verbose = 2, n_jobs = -1)
#rf_random = BaggingClassifier(rfc, max_features = 5, verbose = 2)

#rf_random = RandomizedSearchCV(estimator = bagging, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
best.fit(X_train, y_train)

y_pred = best.predict(X_test)

classificationReport(y_test, y_pred)
accuracyReport(y_test, y_pred)
confusionReport(y_test, y_pred)
rocReport(X_test, y_test, best)
'''