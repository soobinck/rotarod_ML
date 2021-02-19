import os
import string

from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from utils.getDirAbsPath import outputAbsPath

outputDir = os.path.join(outputAbsPath(), 'featureImportance')

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

maxFrames = 6500
nLoops = list(string.ascii_lowercase)
nFeatures = int(maxFrames/2)




importance_columns = ['feature' + str(i) for i in range(1, 11)]
index = [s + str(i) for i in nFeatures for s in list(string.ascii_lowercase)]
acc_mp = pd.DataFrame(
    index=index,
    columns=['accuracy'].append(nFeatures))

for s in nLoops:
    for i in range(1, 11):
        WT = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_WT.csv'
        YAC = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_YAC.csv'

        dfWT = pd.read_csv(WT, index_col=0)
        dfWT = dfWT.iloc[0:i]
        dfWT.dropna(axis=1, how='any', inplace=True)
        dfWT.loc['label'] = 0

        dfYAC = pd.read_csv(YAC, index_col=0)
        dfYAC = dfYAC.iloc[0:i]
        dfYAC.dropna(axis=1, how='any', inplace=True)
        dfYAC.loc['label'] = 1

        df = pd.concat([dfWT, dfYAC], axis=1)

        Xy = df.to_numpy().transpose()
        shuffle(Xy)
        X, y = Xy[:, :-1], Xy[:, -1]

        model = DecisionTreeClassifier()

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.3)
        model.fit(Xtrain, ytrain)
        importance = model.feature_importances_
        y = model.predict(Xtest)
        acc = np.mean(y == ytest)

        acc_mp.loc[s + str(i), 'accuracy'] = acc

        nthFeature = 0
        for imp in importance:
            acc_mp.loc[s+str(i), nthFeature] = imp
            nthFeature += 1

acc_mp.to_csv('./decisionTreeImportanceLoopingDifferenceLengthofWindows.csv')

