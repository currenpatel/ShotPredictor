import numpy as np 
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from featureSelection import *
from eval import *
import time

start_time = time.time()


#MINMIXSCALE
'''
sc = MinMaxScaler()()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)
'''

def findHyperparameters(X_test, y_test, X_train, y_train, start_time):
    #C_range = np.logspace(-4, 4, 5) #1.0 10000
    #gamma_range = np.logspace(-8, 2, 6) #100 100

    #log range
    C_range = np.logspace(-5, 1, 7)# 0.1 0.215
    gamma_range = np.logspace(-3, 1, 5)# 1.0 2.512

    #normal range
    #C_range = np.linspace(0.001, 0.03, num=3)
    #gamma_range = np.linspace(0.75, 1.25, num=3)

    param_grid = dict(gamma=gamma_range, C=C_range)
    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 42)
    svc = svm.SVC(probability = True)
    best = GridSearchCV(svc, param_grid=param_grid, cv=skf, verbose = 2, n_jobs = -1, return_train_score = True)

    #Train the model using the training sets
    best.fit(X_train, y_train)
    paramString = ("Best parameters:\n" + str(best.best_params_) + "\n")
    print(paramString)
    #print("Grid Search Results:\n")
    [print(key, value) for key, value in best.cv_results_.items()]
    
    #scores = best.cv_results_.get('mean_test_score')
    #print(scores)

    plot_grid_search("svm" ,best.cv_results_, gamma_range, C_range, 'Gamma', "C")
    
    #Predict the response for test dataset
    y_pred = best.predict(X_test)

    writeReport(best.cv_results_ ,X_test, X_train, y_test, y_pred ,best, "svm", (time.time() - start_time))
    return best, best.cv_results_

X_train, X_test, y_train, y_test, df = loadData(treeFeatureReduction)

#scale data
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#best = svm.SVC(C = 1.0, gamma = 100, verbose = 2)
best, cv_results_ = findHyperparameters(X_test, y_test, X_train, y_train, start_time)



'''
#data viz
sns.pairplot(df,hue='shot_made_flag',palette='Dark2')
plt.savefig("data viz")
'''