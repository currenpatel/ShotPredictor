import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from featureSelection import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from eval import *
from sklearn import svm


df = pd.read_csv("data/clean.csv")

categories = df.select_dtypes([object]).columns


#one hot
for var in categories:
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], 1)
    df = df.drop(var, 1)

random_sample = df.take(np.random.permutation(len(df))[:3])
#print(random_sample)

X = df.drop('shot_made_flag', 1)
y = np.array(df['shot_made_flag']).astype(int)
#X = treeFeatureReduction(X, y)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
'''
nComponents = 5
pca = PCA(nComponents)
projected = pca.fit_transform(X)
pca.fit(X)
print(pca.explained_variance_ratio_)
'''


nComponents = 2
pca = PCA(nComponents)
pca.fit(X_train)
X_t_train = pca.fit_transform(X_train)
X_t_test = pca.transform(X_test)
clf = svm.SVC(kernel = 'linear')
clf.fit(X_t_train, y_train)
print(pca.explained_variance_ratio_)
y_pred = clf.predict(X_t_test)

print(classificationReport(y_test, y_pred))
print(accuracyReport(y_test, y_pred))
print(confusionReport(y_test, y_pred))

if nComponents == 2:
    principalDf = pd.DataFrame(data = X_t_test, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(y_pred, columns = ['shot_made_flag'])], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['shot_made_flag'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("graphs/" + "pca" + ".png", bbox_inches='tight')