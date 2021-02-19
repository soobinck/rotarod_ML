from dataAnalysis import basicStat
# from dataAnalysis import distribution
from dataAnalysis.stepUP import waitTimeB4StepUpPlot
import numpy as np


def allAnalysis(paths, secondsPerInterval=30, framesPerSecond=20, maxFrames=6500):
    framesPerInterval = framesPerSecond * secondsPerInterval
    maxNumberofIntervals = int(np.ceil(maxFrames / framesPerInterval))

    # basicStat.basicStatAnalysis(paths, secondsPerInterval, framesPerSecond, maxFrames)
    waitTimeB4StepUpPlot(paths, secondsPerInterval, maxNumberofIntervals, framesPerSecond, maxFrames)

userinput_0Dir = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/cleanedAnalyzed/0'
userinput_1Dir = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/cleanedAnalyzed/1'

dir0 = ({'dir': userinput_0Dir, 'label': str(0)})
dir1 = ({'dir': userinput_1Dir, 'label': str(1)})
dirs = [dir0, dir1]

allAnalysis(dirs)
