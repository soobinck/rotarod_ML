import glob
import shutil
from utils.getDirAbsPath import outputAbsPath
import os

def concatenateCSVs(dir_labelDictionaries):
    '''

    :param dir_labelDictionaries:
    :return: The first element in the array is with label of 0 and the second is of 1.
    '''
    prefix = 'concat'
    outputCSVs = []
    if dir_labelDictionaries[0]['label'] == '0':
        inputDir0 = dir_labelDictionaries[0]['dir']
        inputDir1 = dir_labelDictionaries[1]['dir']
    else:
        inputDir0 = dir_labelDictionaries[1]['dir']
        inputDir1 = dir_labelDictionaries[0]['dir']

    allCSVs0 = glob.glob(inputDir0 + '/*.csv')
    allCSVs0.sort()

    allCSVs1 = glob.glob(inputDir1 + '/*.csv')
    allCSVs1.sort()

    allCSVsArray = [allCSVs0, allCSVs1]

    for allCSVs in allCSVsArray:
        count = 0
        outputPath = os.path.join(outputAbsPath(), prefix + str(allCSVsArray.index(allCSVs)) + '.csv')
        outputCSVs.append(outputPath)
        with open(outputPath, 'w') as outputFile:
            for i, inputPath in enumerate(allCSVs):
                with open(inputPath, 'r') as inputFile:
                    if i != 0:
                        inputFile.readline()
                    shutil.copyfileobj(inputFile, outputFile)
                    count += 1
            print('%d files have been combined.' % count)

    return outputCSVs

