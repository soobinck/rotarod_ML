from dataAnalysis.stepUP.stepUpHeightCalculation import calculateStepUpHeight
from dataAnalysis.stepUP.stepUpHeightPlot_singleCSV import stepUpHeightPlot
import os

# userinput_secondsPerInterval = 30
# userinput_framesPerSecond = 20
# userinput_classification = ['WT', 'YAC']
# userinput_0Day3Dir = \
#     '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day3_WT'
# userinput_1Day3Dir = \
#     '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day3_YAC'
# userinput_maxFrames = 6500
#
# dir0 = ({'dir': userinput_0Day3Dir, 'label': 'Day3_WT'})
# dir1 = ({'dir': userinput_1Day3Dir, 'label': 'Day3_YAC'})
# dirs = [dir0, dir1]
# stepUpHeightPlot(calculateStepUpHeight(dirs))



userinput_secondsPerInterval = 30
userinput_framesPerSecond = 20
userinput_classification = ['WT', 'YAC']
userinput_0Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day4_WT'
userinput_1Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day4_YAC'
userinput_maxFrames = 6500

dir0 = ({'dir': userinput_0Day3Dir, 'label': 'Day4_WT'})
dir1 = ({'dir': userinput_1Day3Dir, 'label': 'Day4_YAC'})
dirs = [dir0, dir1]
results = calculateStepUpHeight(dirs)

