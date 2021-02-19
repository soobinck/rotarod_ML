from dataPrep.main_dataPrep import prepareData

userinput_0Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/Day3_WT'
userinput_1Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/data_all/Day3_2and3monthOld_rotarodAnalysis/YAC128'
userinput_maxFrames = 6500

dir0 = ({'dir': userinput_0Day3Dir, 'label': str(0)})
dir1 = ({'dir': userinput_1Day3Dir, 'label': str(1)})
# dirs = [dir0, dir1]
dirs = [dir1]

outputDir = prepareData(dirs)
