from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

WT = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_WT.csv'
YAC = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_YAC.csv'

dfWT = pd.read_csv(WT, index_col=0)
dfWT = dfWT.iloc[0:3]
dfWT.dropna(axis=1, how='any', inplace=True)
dfWT.loc['label'] = 0


dfYAC = pd.read_csv(YAC, index_col=0)
dfYAC = dfYAC.iloc[0:3]
dfYAC.dropna(axis=1, how='any', inplace=True)
dfYAC.loc['label'] = 1

df = pd.concat([dfWT, dfYAC], axis=1)

df = df.to_numpy().transpose()
shuffle(df)
X, y = df[:, :-1], df[:, -1]

model = DecisionTreeClassifier()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.3)
model.fit(Xtrain, ytrain)
importance = model.feature_importances_
y = model.predict(Xtest)
acc = np.mean(y == ytest)

