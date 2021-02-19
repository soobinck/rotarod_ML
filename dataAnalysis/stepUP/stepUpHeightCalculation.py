import os
import shutil

import numpy as np
import pandas as pd

from utils.getDirAbsPath import outputAbsPath, getLastDirectory


#
# if os.path.exists(output):
#     shutil.rmtree(output)
# os.mkdir(output)


def calculateStepUpHeight(dir_labelDictionaries, framesPerSecond=20, secondsPerInterval=30, estimatedMaxFrames=6500):
    csvs = []
    for dir_labelDictionary in dir_labelDictionaries:
        inputDir = dir_labelDictionary['dir']
        inputLabel = dir_labelDictionary['label']

        prefix = 'stepUpHeightRight_'

        framesPerInterval = framesPerSecond * secondsPerInterval
        framesPerMinute = framesPerSecond * 60
        minutePerInterval = framesPerInterval / framesPerMinute
        n = int(np.ceil(estimatedMaxFrames / framesPerInterval))

        dfInd = []
        for i in range(n):
            dfInd.append(i)

        dfRightStepUpHeight = pd.DataFrame(
            index=dfInd
        )

        for roots, dirs, files in os.walk(inputDir):
            for inputFile in files:
                if inputFile.startswith('.') or not inputFile.endswith('.csv'):
                    continue

                dfRightStepUpHeight[inputFile] = np.nan
                df = pd.read_csv(os.path.join(inputDir, inputFile), index_col=0)
                col = 'rel RightY mm'
                d = df[col].diff()
                m = d.lt(0)
                b = (~m).cumsum()
                s = d.mask(~m).abs().groupby(b).transform('sum')
                df['right foot step up height'] = pd.DataFrame(
                    np.select([~b.duplicated(keep='last') & m, d.eq(0)], [s, '1e3'], ''))

                dfs = df['right foot step up height']
                dfLen = len(dfs)
                for i in range(n):
                    start = i * framesPerInterval
                    end = (i + 1) * framesPerInterval
                    if end < len(dfs):
                        splits = dfs[start: end]
                    elif start < len(dfs):
                        splits = dfs[i * framesPerInterval:]
                    else:
                        break
                    splits = splits.replace('', '0')
                    splits = splits.astype(np.float)
                    splits1 = splits != 0
                    nSplitStepUp = np.sum(splits1)
                    if nSplitStepUp == 0:
                        dfRightStepUpHeight.loc[i, inputFile] = np.nan
                    else:
                        splitStepUpHeightmm = np.sum(splits)
                        dfRightStepUpHeight.loc[i, inputFile] = splitStepUpHeightmm / nSplitStepUp

        outputDir = os.path.join(outputAbsPath(), 'stepHeightCalculation')
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        inputDirLastPath = getLastDirectory(inputDir)
        outputCSVPath = os.path.join(outputDir, prefix + inputDirLastPath + '.csv')
        dfRightStepUpHeight.to_csv(outputCSVPath)
        csvs.append(outputCSVPath)
    return csvs
