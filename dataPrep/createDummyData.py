# Create dummy cleanedAnalyzedData in the format of DeepLabCut result
import os
import random
import shutil

import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint

dpath = os.path.join('.', 'cleanedAnalyzedData')
if not os.path.exists(dpath):
    os.mkdir(dpath)

row1 = ['scorer', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000',
        'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000', 'DeepCut_resnet50_rotarod3Jul17shuffle1_1030000'
        ]

row2 = ['bodyparts', 'Rightpaw', 'Rightpaw', 'Rightpaw', 'Leftpaw', 'Leftpaw', 'Leftpaw', 'Tailbase',
        'Tailbase', 'Tailbase', 'Rotarodtop', 'Rotarodtop', 'Rotarodtop', 'Rotarodbottom',
        'Rotarodbottom', 'Rotarodbottom']

row3 = ['coords', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood', 'x',
        'y', 'likelihood']

rowList = [row1, row2, row3]

rows2and3 = []

for i in range(len(row2)):
    rows2and3.append(row2[i] + ' ' + row3[i])


def createDummyData(classification):
    fpaths = []
    for cls in classification:
        fpath = os.path.join('.', 'cleanedAnalyzedData', 'dummy' + cls + '_CLS' + str(classification.index(cls)))
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
        os.mkdir(fpath)
        fpaths.append(fpath)
        n = random.randint(2, 5)
        for ith in range(n):
            fileName = cls + str(ith)
            colDict = {}
            colLen = random.randint(3000, 6000)
            for iCol, col in enumerate(rows2and3):
                if 'bodyparts' in col:
                    colDict[row1[iCol] + str(iCol)] = [row2[iCol], row3[iCol]] + list(range(colLen))
                elif 'likelihood' in col:
                    newCol = [row2[iCol], row3[iCol]] + list(rand(colLen))
                    colDict[row1[iCol] + str(iCol)] = newCol
                elif ('x' in col) or ('y' in col):
                    newCol = [row2[iCol], row3[iCol]] + list(randn(colLen) + randint(100, 300, 1)[0])
                    colDict[row1[iCol] + str(iCol)] = newCol
            df = vars()[fileName] = pd.DataFrame(colDict)
            df.to_csv(os.path.join(fpath, cls + str(ith) + '.csv'), index=False)

    return fpaths
