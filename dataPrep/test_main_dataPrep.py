### Day 3 ###
# import os
#
# # from dataAnalysis.allAnalysis import allAnalysis
# from dataPrep.main_dataPrep import prepareData
# from utils.getDirAbsPath import outputAbsPath
#
# classWT, classYAC = 0, 1
# # TODO: Do not assume the names of columns (ex. Rightpaw x, Rotarod top). Ask for them to the user.
# # TODO: Create `output` dir main project dir
# userinput_secondsPerInterval = 30  # TODO: Ask for interval length.
# userinput_framesPerSecond = 20  # TODO: Ask for frames per second
# userinput_classification = ['WT', 'YAC']  # TODO: Ask which one should be 0 and 1.
# userinput_0Day3Dir = \
#     '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/data_all/Day3_2and3monthOld_rotarodAnalysis/WT'  # TODO: User input
# userinput_1Day3Dir = \
#     '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/data_all/Day3_2and3monthOld_rotarodAnalysis/YAC128'  # TODO: User input
# userinput_maxFrames = 6500
#
# dir0 = ({'dir': userinput_0Day3Dir, 'label': 'Day3_WT'})
# dir1 = ({'dir': userinput_1Day3Dir, 'label': 'Day3_YAC'})
# dirs = [dir0, dir1]
#
# outputDirAbsolutePath = outputAbsPath()
#
# if not os.path.exists(outputDirAbsolutePath):
#     os.mkdir(outputDirAbsolutePath)
#
# dirs = prepareData(dirs)
# # allAnalysis(dirs, userinput_secondsPerInterval, userinput_framesPerSecond, userinput_maxFrames)
#


### Day 4 ###
import os

# from dataAnalysis.allAnalysis import allAnalysis
from dataPrep.main_dataPrep import prepareData
from utils.getDirAbsPath import outputAbsPath

classWT, classYAC = 0, 1
# TODO: Do not assume the names of columns (ex. Rightpaw x, Rotarod top). Ask for them to the user.
# TODO: Create `output` dir main project dir
userinput_secondsPerInterval = 30  # TODO: Ask for interval length.
userinput_framesPerSecond = 20  # TODO: Ask for frames per second
userinput_classification = ['WT', 'YAC']  # TODO: Ask which one should be 0 and 1.
userinput_0Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_ML10/data_all/Day3_2and3monthOld_rotarodAnalysis/WT'  # TODO: User input
userinput_1Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_ML10/data_all/Day3_2and3monthOld_rotarodAnalysis/YAC128'  # TODO: User input
userinput_0Day4Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_ML10/data_all/Day4_2and3monthOld_rotarodAnalysis/WT'  # TODO: User input
userinput_1Day4Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_ML10/data_all/Day4_2and3monthOld_rotarodAnalysis/YAC128'  # TODO: User input

userinput_maxFrames = 6500

dir0day3 = ({'dir': userinput_0Day3Dir, 'label': 'Day3_WT'})
dir1day3 = ({'dir': userinput_1Day3Dir, 'label': 'Day3_YAC'})
dir0day4 = ({'dir': userinput_0Day4Dir, 'label': 'Day4_WT'})
dir1day4 = ({'dir': userinput_1Day4Dir, 'label': 'Day4_YAC'})


dirs = [dir0day3, dir1day3, dir0day4, dir1day4]

outputDirAbsolutePath = outputAbsPath()

if not os.path.exists(outputDirAbsolutePath):
    os.mkdir(outputDirAbsolutePath)

dirs = prepareData(dirs)
# allAnalysis(dirs, userinput_secondsPerInterval, userinput_framesPerSecond, userinput_maxFrames)
