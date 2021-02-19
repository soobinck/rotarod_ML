# TODO: Complete a custom error when abs(mmPERpx1 - mmPERpx0) < threshold in pixel2mm.py


def similarRatios(mmPERpx0, mmPERpx1, threshold):
    assert (abs(mmPERpx1 - mmPERpx0) < threshold)


def isCSVFile(inputFile: str):

    return (not inputFile.startswith('.')) and inputFile.endswith('.csv')


    # if not os.path.exists(inputDir):
    #     raise Exception(
    #         'The folder \'%s\' does not exist. Double check that the folder exists or input its absolute path.' % inputDir)
