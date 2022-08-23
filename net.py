#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
import numpy as np 
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from featureSelection import *
from eval import *
import time

start_time = time.time()

#need to normalize

X_train, X_test, y_train, y_test, df = loadData(treeFeatureReduction)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# baseline model
def create_baseline():
    classifier = Sequential()
    input_dim = layerNum = X_train.shape[1]
    #First Hidden Layer
    layerNum = layerNum // 2
    classifier.add(Dense(layerNum, activation='relu', kernel_initializer='random_normal', input_dim=input_dim))
    #Second  Hidden Layer
    layerNum = layerNum // 2
    classifier.add(Dense(layerNum, activation='relu', kernel_initializer='random_normal'))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    #classifier.summary()
    # compile the keras model
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return classifier
# evaluate model with standardized dataset
net = KerasClassifier(build_fn=create_baseline)
kfold = StratifiedKFold(n_splits=3, shuffle=True)
batch_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
epochs = [10, 20, 30, 40, 50]
grid = dict(batch_size=batch_size, epochs=epochs)
best = GridSearchCV(estimator=net, param_grid=grid, cv=kfold, return_train_score = True)
best = best.fit(X_train, y_train)
y_pred = best.predict(X_test)

plot_grid_search("net" ,best.cv_results_, batch_size, epochs, 'Batch Size', "Epoch")
writeReport(best.cv_results_ ,X_test, X_train, y_test, y_pred ,best, "net", (time.time() - start_time))



paramString = ("Best parameters:\n" + str(best.best_params_) + "\n")
print(paramString)
#print("Grid Search Results:\n")
[print(key, value) for key, value in best.cv_results_.items()]

print(classificationReport(y_test, y_pred))
print(accuracyReport(y_test, y_pred))
print(confusionReport(y_test, y_pred))

'''
# define the keras model
classifier = Sequential()
input_dim = layerNum = X.shape[1]
#First Hidden Layer
layerNum = layerNum // 2
classifier.add(Dense(layerNum, activation='relu', kernel_initializer='random_normal', input_dim=input_dim))
#Second  Hidden Layer
layerNum = layerNum // 2
classifier.add(Dense(layerNum, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(layerNum, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# compile the keras model
classifier.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# fit the keras model on the dataset
classifier.fit(X_train, y_train, epochs=50, batch_size=10, verbose = 1)
# evaluate the keras model
_, accuracy = classifier.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
y_pred = classifier.predict(X_test)
y_pred =(y_pred>0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
'''