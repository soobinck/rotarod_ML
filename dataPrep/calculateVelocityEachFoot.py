import copy
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def addVelocityColumnsBothFeet(inputDir):
    prefix = 'ad_'
    outputDir = os.path.join('/', *inputDir.split('/')[:-1], 'addVelocityColumnsBothFeet')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    for roots, dirs, files in os.walk(inputDir):
        for inputFile in files:
            df_org = pd.read_csv(os.path.join(inputDir, inputFile))
            df = copy.deepcopy(df_org)

            PawsColumnNames = [{'name': 'Rightpaw', 'x': 'Rightpaw x', 'y': 'Rightpaw y'},
                               {'name': 'Leftpaw', 'x': 'Leftpaw x', 'y': 'Leftpaw y'}]

            for pawColumnDictionary in PawsColumnNames:
                x = pawColumnDictionary['x']
                y = pawColumnDictionary['y']
                name = pawColumnDictionary['name']
                delx = np.diff(df[x])
                dely = np.diff(df[y])
                signs = np.sign(dely)

                squaredDelx = pow(delx, 2)
                squaredDely = pow(dely, 2)
                sums = squaredDelx + squaredDely

                diff = np.sqrt(sums) * signs * -1

                df[name + ' ' + 'euclidean velocity'] = np.append(np.nan, diff)

                outputFile = os.path.join(outputDir, prefix + inputFile)

                df.to_csv(outputFile)

    shutil.rmtree(inputDir)
    return outputDir
