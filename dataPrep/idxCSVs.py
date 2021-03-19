import os
import shutil

import numpy as np
import pandas as pd


def idxCSVs(inputDir):
    i = 0
    prefix = i
    outputDir = os.path.join('/', *inputDir.split('/')[:-1], 'idxCSV')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    files = next(os.walk(inputDir))[2]

    for inputFile in files:
        if (not inputFile.startswith('.')) and (inputFile.endswith('.csv')):
            df = pd.read_csv(os.path.join(inputDir, inputFile), index_col=0)

            outputFile = os.path.join(outputDir, str(prefix) + 'idx_' + inputFile)
            df.to_csv(outputFile)
            print('Indexed files saved in %s.' % outputDir)
            prefix += 1

    shutil.rmtree(inputDir)

    return outputDir
