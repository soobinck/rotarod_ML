# Guided by https://machinelearningmastery.com/calculate-feature-importance-with-python/


from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

WT = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_WT.csv'
YAC = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_YAC.csv'

dfWT = pd.read_csv(WT, index_col=0)

dfYAC = pd.read_csv(YAC, index_col=0)
dfWT.loc['label'], dfYAC.loc['label'] = [0, 1]

df = pd.concat([dfWT, dfYAC], axis=1)

df = df.to_numpy().transpose()
shuffle(df)
X, y = df[:, :-1], df[:, -1]

model = DecisionTreeClassifier()

X0 = X[:, 0:2]
X0train, X0test, y0train, y0test = train_test_split(X0, y, test_size=0.3)
model.fit(X0train, y0train)
importance = model.feature_importances_
y0 = model.predict(X0test)
acc = np.mean(y0 == y0test)

X0 = X[:, 0:3]


