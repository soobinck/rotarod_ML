import os


def dataAbsPath(relativePath=os.path.join('./')):
    absolutePathofRelativePath = os.path.abspath(relativePath).split('/')

    return os.path.join('/',
                        *absolutePathofRelativePath[:absolutePathofRelativePath.index('rotarod_ML') + 1])


def outputAbsPath(relativePath=os.path.join('./')):
    absolutePathofRelativePath = os.path.abspath(relativePath).split('/')

    return os.path.join('/',
                        *absolutePathofRelativePath[:absolutePathofRelativePath.index('rotarod_ML') + 1],
                        'output')


def getLastDirectory(inputDir):
    if inputDir.endswith('/'):
        inputDir = inputDir[-1]
    return os.path.split(inputDir)[-1]
