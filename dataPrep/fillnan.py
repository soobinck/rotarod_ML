import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fillnan(userinput_columns_likelihoods, userinput_pBoudn,inputDir):
    '''

    :param inputDir:
    :return:
    '''
    prefix = 'fi_'
    outputDir = os.path.join('/', *inputDir.split('/')[:-1], 'fillnan')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    for roots, dirs, files in os.walk(inputDir):
        for inputFile in files:
            df = pd.read_csv(os.path.join(inputDir, inputFile))

            for userinput_columns_likelihood in userinput_columns_likelihoods:
                col = userinput_columns_likelihood['column']
                likelihood = userinput_columns_likelihood['likelihood']
                nanIndex = np.array(df.index[df[likelihood] < userinput_pBoudn])

                df.loc[nanIndex, col] = np.nan

                for i in nanIndex:
                    if i == 0:
                        continue

                    if i >= len(df) - 1:
                        break

                    prevIndex = i - 1
                    postIndex = i + 1

                    while any(np.isnan(df.loc[prevIndex, col])):
                        if prevIndex == 0:
                            break
                        else:
                            prevIndex -= 1
                    if prevIndex == 0:
                        continue

                    prevs = df.loc[prevIndex, col]

                    if postIndex >= len(df) - 1:
                        break

                    while any(np.isnan(df.loc[postIndex, col])):
                        postIndex += 1
                        if postIndex == len(df) - 1:
                            break
                    if postIndex == len(df) - 1:
                        break

                    posts = df.loc[postIndex, col]

                    steps = postIndex - prevIndex
                    stepDiffs = (posts - prevs) / steps

                    df.at[i, col] = prevs + stepDiffs

            outputFile = os.path.join(outputDir, prefix + inputFile)

            df.to_csv(outputFile)

    shutil.rmtree(inputDir)
    return outputDir
