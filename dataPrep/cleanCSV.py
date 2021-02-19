import csv
import os
import shutil

from utils.deleteCommonWordsInFileName import deleteCommonWordsInFileName
from utils.getDirAbsPath import outputAbsPath

# inputDirs = ['cleanedAnalyzedData/dummyWT_CLS0', 'cleanedAnalyzedData/dummyYAC_CLS1']


def cleanCSV(inputDir):
    prefix = 'cl_'
    outputDir = os.path.join(outputAbsPath(), 'cleanCSV')

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)

    if '/' in inputDir[-1]:
        inputDir = inputDir[-1]

    for root, dirs, files in os.walk(inputDir):
        for file in files:
            if (not file.startswith('.')) and file.endswith('.csv'):
                with open(os.path.join(inputDir, file), 'r') as readFile:
                    csvReader = csv.reader(readFile)

                    # Delete the first row and join the second and third rows
                    for i, row in enumerate(csvReader):
                        if i == 0:
                            continue
                        if i == 1:
                            row1 = row
                        if i == 2:
                            row2 = row
                            break
                    row0ToBe = zip(row1, row2)
                    row0ToBe = tuple(row0ToBe)
                    row0Col = []
                    for tup in row0ToBe:
                        newCol = ' '.join(tup)
                        row0Col.append(newCol)
                    file = deleteCommonWordsInFileName(file)
                    outputFileName = os.path.join(outputDir, prefix + file)

                    with open(outputFileName, 'w') as outputFile:
                        csvWriter = csv.writer(outputFile)
                        csvWriter.writerow(row0Col)
                        for line in readFile:
                            outputFile.write(line)

    return outputDir
