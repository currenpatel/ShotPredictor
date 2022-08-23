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
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

def kBestReduction(X, y, n = 10):
    #apply SelectKBest class to extract top n best features
    bestfeatures = SelectKBest(f_classif, k=n)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    largest = featureScores.nlargest(n,'Score')
    #print(largest)  #print n best features
    return X[largest[largest.columns[0]].values.tolist()]

#tree feature reduction
def treeFeatureReduction(X, y, n = 15):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    largest = feat_importances.nlargest(n)
    #largest.plot(kind='barh')
    #plt.show()

    return X[largest.index.values.tolist()]
    

#recursive feature reduction
@ignore_warnings(category=ConvergenceWarning)
def recursiveFeatureReduction(X, y, n = 10):
    model = LogisticRegression(solver='lbfgs', max_iter = 10000)
    rfecv = RFECV(model, step = 5, cv = 3, n_jobs = -1, min_features_to_select = n)
    rfecv = rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print((X.loc[:, rfecv.support_]))
    return X.loc[:, rfecv.support_]
