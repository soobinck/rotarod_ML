import os

import pandas as pd

def MLAnalysis():
    pass

inputDir = './skmodels_acc'
outputDir = './output'
#TODO: flase negative, false positive, true negative, true positive
accs = pd.DataFrame(columns=[
    'Nearest Neighbors',
    'Linear SVM',
    'RBF SVM',
    'Gaussian Process',
    'Decision Tree',
    'Random Forest',
    'Neural Net',
    'AdaBoost',
    'Naive Bayes',
    'QDA']
)
for roots, dirs, files in os.walk(inputDir):
    for inputFile in files:
        if 'testAcc' in inputFile:
            df = pd.read_csv(os.path.join(inputDir, inputFile), header=None)
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df[1:]
            accs = pd.concat([accs, df], ignore_index=True)

accs.to_csv('./output/accuracies.csv')
