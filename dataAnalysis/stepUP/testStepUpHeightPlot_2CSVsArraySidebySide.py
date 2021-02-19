from dataAnalysis.stepUP.stepUpHeightPlot_2CSVsArraySidebySide import stepUp2HeightPlot

path0 = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_WT.csv'
path1 = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/stepHeightCalculation/stepUpHeightRight_Day4_YAC.csv'

dir0 = ({'dir': path0, 'label': 'Day4_WT'})
dir1 = ({'dir': path1, 'label': 'Day4_YAC'})
dirs = [dir0, dir1]

dirs = [dir0, dir1]

stepUp2HeightPlot(dirs)
