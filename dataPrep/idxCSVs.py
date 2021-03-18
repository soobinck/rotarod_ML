import os
import shutil

import numpy as np
import pandas as pd

def idxCSVs(inputDir):
    i = 0
    prefix = i
    outputDir = os.path.join('/', *inputDir.split('/')[:-1], 'idxCSVs')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    for roots, dirs, files in os.walk(inputDir):
        for inputFile in files:
            df = pd.read_csv(os.path.join(inputDir, inputFile))

            df['step up'] = df['Rightpaw euclidean velocity'] >= 0

            steppedUp = df.index[df['step up']]
            waitTime = np.append(
                steppedUp[0],
                steppedUp[1:] - steppedUp[:-1] - 1
            )

            df['wait time b4 step up'] = 0
            df.loc[df['step up'] == False, 'wait time b4 step up'] = 0

            inxs = df.index[df['step up'] != False]

            for k, i in enumerate(inxs):
                if k == 0:
                    df.loc[i, 'wait time b4 step up'] = df.index[i]
                else:
                    df.loc[i, 'wait time b4 step up'] = df.index[i] - df.index[inxs[k - 1]] - 1

            outputFile = os.path.join(outputDir, str(prefix) + inputFile)
            df.to_csv(outputFile)
            prefix += 1

    shutil.rmtree(inputDir)
    return outputDir
