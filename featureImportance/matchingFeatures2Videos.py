import os
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.getDirAbsPath import outputAbsPath
import os
import string

from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot
import pandas as pd
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from utils.getDirAbsPath import outputAbsPath

outputDir = os.path.join(outputAbsPath(), 'featureImportance')
framesPerInterval = 200
maxFrames = 6500

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

nLoops = range(100)
nFeatures = range(1, int(maxFrames / framesPerInterval))

importance_columns = ['feature' + str(i) for i in nFeatures]
index = [str(i) for i in nFeatures]
acc_mp = pd.DataFrame(
    index=nLoops,
    columns=['accuracy'].append(nFeatures))

for s in nLoops:
    WT = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day3_WT/st_ad_mm_fi_cl_190623_Day3_146m6_rotarod1_2Nov15.csv'
    YAC = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day3_YAC/st_ad_mm_fi_cl_190629_Day3_145m3_rotarod1_2Nov15.csv'

    dfWT = pd.read_csv(WT, index_col=0)['Rightpaw y']
    lastRowConsidered = np.floor(len(dfWT) / framesPerInterval) * framesPerInterval
    dfWT = dfWT.loc[0: lastRowConsidered - 1]

    dfWT = pd.DataFrame(dfWT.values.reshape(int(len(dfWT) / framesPerInterval), framesPerInterval),
                        columns=range(framesPerInterval))
    dfWT.dropna(axis=0, how='any', inplace=True)
    dfWT['label'] = 0

    dfYAC = pd.read_csv(YAC, index_col=0)['Rightpaw y']
    lastRowConsidered = np.floor(len(dfYAC) / framesPerInterval) * framesPerInterval
    dfYAC = dfYAC.loc[0: lastRowConsidered - 1]

    dfYAC = pd.DataFrame(dfYAC.values.reshape(int(len(dfYAC) / framesPerInterval), framesPerInterval),
                         columns=range(framesPerInterval))
    dfYAC.dropna(axis=0, how='any', inplace=True)
    dfYAC['label'] = 1

    # Trim the dataframes so that both of them have the same duration.
    # Head:
    if dfWT.index[0] < dfYAC.index[0]:
        dfWT = dfWT.loc[dfYAC.index[0]:]
    else:
        dfYAC = dfYAC.loc[dfWT.index[0]:]

    # Tail
    if dfWT.index[-1] > dfYAC.index[-1]:
        dfWT = dfWT.loc[:dfYAC.index[-1]]
    else:
        dfYAC = dfYAC.loc[:dfWT.index[-1]]

    df = pd.concat([dfWT, dfYAC])
    meanWT = np.mean(dfWT.values, axis=1)
    meanWT_label = np.hstack((meanWT, np.zeros(1)))
    meanYAC = np.mean(dfYAC.values, axis=1)
    meanYAC_label = np.hstack((meanYAC, np.ones(1)))

    Xy = np.vstack((meanWT_label, meanYAC_label))
    X, y = Xy[:, :-1], Xy[:, -1]
    model = AdaBoostClassifier()

    Xtrain, Xtest, ytrain, ytest = X, X, y, y
    model.fit(Xtrain, ytrain)
    importance = model.feature_importances_
    y = model.predict(Xtest)
    acc = np.mean(y == ytest)

    acc_mp.loc[s, 'accuracy'] = acc

    nthFeature = 0
    for imp in importance:
        acc_mp.loc[s, nthFeature] = imp
        nthFeature += 1

acc_mp.loc['Total'] = acc_mp.sum(numeric_only=True)
mostImportantInterval = np.argsort(acc_mp.loc['Total']).loc[-2] - 1  # -1 because accuracy is index 0.
acc_mp.to_csv(os.path.join(outputAbsPath(), 'featureImportance', 'featureImportance_adaBoost_individualFiles.csv'))

# t = 'The most important interval is from % i to % i seconds.' % mostImportantInterval
