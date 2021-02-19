import os
import shutil
import numpy as np
import pandas as pd
from utils.calc import mean
from utils.getDirAbsPath import outputAbsPath

def intervalsSplit(inputDir, col, fun, framesPerInterval=30,
                   framesPerSecond=20, maxFrames=6000):
    '''
    :param inputDir: Path to a directory that you wish to perform interval split analysis of with `fun(ction)`.
                     It does NOT accpet arrays of paths.
                     Must be a string to a path.
    :param col: The name of the column that you want to perform the analysis on.
    :param fun: The type of analysis that you want to perform the analysis of.
                It can be `mean` and more functionalities to come. TODO
                For example, `mean` returns calculates the mean of the each interval.
    :param framesPerInterval: Number of frames per interval.
    TODO: add a param secondsPerInterval.
    :param framesPerSecond: Number of frames per one second.
    :param maxFrames: The maximum number of frames(rows) that can be possibly in any of the experiments (csv files).
    :return:
            Return
            [
             [start of the interval ~ end of the interval (in frames), fun(values in the first interval)],
             [start of the interval ~ end of the interval (in frames), fun(values in the second interval)],
             ...
             [start of the interval ~ end of the interval (in frames),
              fun(values in the last=ceiling(maxFrames/framesPerInterval) interval)]
            ],

            and saves this array in rotarod_ML/output/interval/`name of the given directory` + `func`.csv

            (Does not return but) Saves
            [
             [previous function necessary for the fun(values in the first interval), the number of values in the first interval],
             [previous function necessary for the fun(values in the second interval), the number of values in the second interval],
             ...
             [previous function necessary for the fun(values in the last=ceiling(maxFrames/framesPerInterval) interval), the number of values in the last interval]
            ]
            in a separate csv files for each experiments in rotarod_ML/output/interval/`name of the experiment` + `func`.csv
    '''
    inputDir = os.path.abspath(inputDir)
    outputDirAbsolutePath = outputAbsPath()

    outputDirInterval = os.path.join(outputDirAbsolutePath, 'interval')
    if not os.path.exists(outputDirInterval):
        os.mkdir(outputDirInterval)

    outputDirIndividuals = (
        os.path.join(outputDirAbsolutePath, 'interval/%sIndividual' % (fun.__name__ + inputDir.split('/')[-1])))
    if not os.path.exists(outputDirIndividuals):
        os.mkdir(outputDirIndividuals)

    outputDirAll = os.path.join(outputDirAbsolutePath, 'interval/%s' % (fun.__name__ + inputDir.split('/')[-1]))
    if not os.path.exists(outputDirAll):
        os.mkdir(outputDirAll)

    framesPerMinute = framesPerSecond * 60
    MinutePerInterval = framesPerInterval / framesPerMinute
    n = int(np.ceil(maxFrames / framesPerInterval))

    dfIndex = []

    for i in range(n):
        dfIndex.append(str(i * framesPerInterval) + '~' + str((i + 1) * framesPerInterval) + '(frames)')

    dfFunAll = pd.DataFrame(
        index=dfIndex
    )

    dfIndividual = pd.DataFrame(
        index=dfIndex
    )

    dfIndividual['valuesSum'] = np.zeros(n)
    dfIndividual['numberofFramesSum'] = np.zeros(n)
    totalValuesSum_totalN = np.zeros([n, 2])
    for roots, dirs, files in os.walk(inputDir):
        for inputFile in files:
            df = pd.read_csv(os.path.join(inputDir, inputFile), index_col=0)
            dfFunAll[inputFile] = ''
            dfCol = df[col]
            dfLen = len(df)
            for i in range(n):
                start = i * framesPerInterval
                end = (i + 1) * framesPerInterval
                if end > dfLen:
                    dfSelected = dfCol[start:].values
                elif end < dfLen:
                    dfSelected = dfCol[start:end].values
                else:
                    raise ('The number of rows in %s is greater than the given maximum number of frames.' % inputFile)

                indexName = str(i * framesPerInterval) + '~' + str((i + 1) * framesPerInterval) + '(frames)'

                selectedValuesSum = np.sum(dfSelected)  # TODO: Make this a part of fun (mean).
                dfIndividual.loc[indexName, 'valuesSum'] = selectedValuesSum  # TODO: Make this a part of fun (mean).
                dfIndividual.loc[indexName, 'numberofFramesSum'] = len(
                    dfSelected)  # TODO: Make this a part of fun (mean).
                totalValuesSum_totalN[i, 0] += selectedValuesSum
                totalValuesSum_totalN[i, 1] += len(dfSelected)
            dfIndividual.to_csv(
                os.path.join(outputDirIndividuals, '%s' % (fun.__name__ + inputFile.split('/')[-1])),
                mode='w+')
    nonZeroIndex = np.where(totalValuesSum_totalN[:, 1] != 0)[0]
    dfFunAll = np.zeros(n).transpose()
    dfFunAll[nonZeroIndex] = (totalValuesSum_totalN[nonZeroIndex, 0] / totalValuesSum_totalN[nonZeroIndex, 1])
    dfFunAll = pd.DataFrame(dfFunAll)
    dfFunAll.to_csv(os.path.join(outputDirAll, inputDir.split('/')[-1] + '.csv'), mode='w+')
    return dfFunAll

# intervalsSplit(os.path.join('..', 'cleanedAnalyzedData/prepped_dummyWT_CLS0'), 'Rightpaw x', mean)
