import os
import pickle
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.throwError import similarRatios

inputDir = os.path.join('..', 'cleanedAnalyzedData/prepped_dummyWT_CLS0')
inputDir_YAC = 'cleanedAnalyzedData/prepped_dummyYAC_CLS0'

outputDir = os.path.join('..', 'cleanedAnalyzedData/mm_dec2020_stepupIntv_rightVel_normalizedX_nanfilled_cleaned')
outputDir_YAC = '../cleanedAnalyzedData/mm_dec2020_stepupIntv_rightVel_normalizedX_nanfilled_cleaned_YAC'

# function inputs
# inputDirs = ['cleanedAnalyzedData/prepped_dummyWT_CLS0', 'cleanedAnalyzedData/prepped_dummyYAC_CLS1']
inputDirs = [os.path.join('..', 'cleanedAnalyzedData/prepped_dummyWT_CLS0')]

width_mm = 57
height_mm = 30
# end of function inputs


rtop = 'Rotarodtop'


def pixel2mm(inputDir):
    prefix = 'mm_'
    outputDir = os.path.join('/', *inputDir.split('/')[:-1], 'pixel2mm')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    for roots, dirs, files in os.walk(inputDir):
        for inputFile in files:
            df = pd.read_csv(os.path.join(inputDir, inputFile), index_col=0)
            if all(any(x in s for s in df.columns) for x in ['Rotarodtop', 'Rotarodbottom']):
                topRight = pd.DataFrame({
                    'x': df['Rotarodtop x'],
                    'y': df['Rotarodtop y'],
                    'likelihood': df['Rotarodtop likelihood']
                })
                bottomLeft = pd.DataFrame({
                    'x': df['Rotarodbottom x'],
                    'y': df['Rotarodbottom y'],
                    'likelihood': df['Rotarodbottom likelihood']
                })

                xTopRight = np.mean(topRight.loc[topRight['likelihood'] > 0.99, 'x'])
                yTopRight = np.mean(topRight.loc[topRight['likelihood'] > 0.99, 'y'])

                xBottomLeft = np.mean(bottomLeft.loc[bottomLeft['likelihood'] > 0.99, 'x'])
                yBottomLeft = np.mean(bottomLeft.loc[bottomLeft['likelihood'] > 0.99, 'y'])

                height_px = yBottomLeft - yTopRight
                width_px = xTopRight - xBottomLeft

                mmPERpx0 = height_mm / height_px
                mmPERpx1 = width_mm / width_px

                # TODO: uncomment the threshold and delete 'threshold = 1000'
                # threshold = (mmPERpx1 + mmPERpx0) / 2 * 0.1
                threshold = 1000

                try:
                    similarRatios(mmPERpx0, mmPERpx1, threshold)
                except AssertionError:
                    print('\n  Skipping the file %s' % inputDir + '/' + inputFile,
                          'The camera might not be level or the distance given might not be accurate. \nThe height derived %.2fmm/pixel but the width drived %.2fmm/pixel. \nThe threshold was calculated %.2fmm/pixel.\n' % (
                              mmPERpx0, mmPERpx1, threshold))

                df['rel RightY mm'] = (df['Rightpaw y'] - yTopRight) * mmPERpx0
                df['rel LeftY mm'] = (df['Leftpaw y'] - yTopRight) * mmPERpx0

                x0_rel = df.loc[~np.isnan(df['Leftpaw x']), 'Leftpaw x'].iloc[0]

                df['rel LeftX px'] = df['Leftpaw x'] - x0_rel
                df['rel RightX px'] = df['Rightpaw x'] - x0_rel
                df['rel LeftX mm'] = df['rel LeftX px'] * mmPERpx0
                df['rel RightX mm'] = df['rel RightX px'] * mmPERpx0

                outputFile = prefix + inputFile

                df.to_csv(os.path.join(outputDir, outputFile))
    shutil.rmtree(inputDir)
    return outputDir
