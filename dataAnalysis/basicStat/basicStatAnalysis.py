import os
import pandas as pd
from utils.intervalsSplit import intervalsSplit
from utils.calc import mean


def basicStatAnalysis(paths, col, func, framesPerInterval=30, framesPerSecond=20, maxFrames=6000):
    '''
    :param paths: an array of directories.
    :param col: The column name to implement analysis (fun) on.
    :param func: A predefined function that perform analysis.
    :param framesPerInterval: Number of frames per an interval.
    :param framesPerSecond: Number of frames per one second.
    :param maxFrames: The maximum number of frames(rows) that can be possibly in any of the experiments (csv files).

    :return:
           [
             [start of the interval ~ end of the interval (in frames), fun(values in the first interval)],
             [start of the interval ~ end of the interval (in frames), fun(values in the second interval)],
             ...
             [start of the interval ~ end of the interval (in frames),
              fun(values in the last=ceiling(maxFrames/framesPerInterval) interval)]
            ],

            As saved the return value in rotarod_ML/output/interval/`name of the given directory` + `func`.csv

    '''
    result = []
    # `result` should be a list of 2 lists.
    # 2 lists individually represents func analysis of all examples of each genotype.
    for path in paths:
        label = path[-1]
        means = intervalsSplit(path, col, func, framesPerInterval, framesPerSecond, maxFrames)
        result.append(means)
    # TODO: Visualize the results and save to .pngs.

    return result

# paths = [os.path.abspath(os.path.join('..', 'output/cleanedAnalyzedData/0')), os.path.abspath(os.path.join('..', 'output/cleanedAnalyzedData/1'))]
# result = basicStatAnalysis(paths, 'Rightpaw y', mean)
