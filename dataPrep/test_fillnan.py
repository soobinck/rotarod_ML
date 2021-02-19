from dataPrep.fillnan import fillnan


userinput_classification = ['WT', 'YAC']  # TODO: Ask which one should be 0 and 1.
inputDir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/0'
userinput_1Day3Dir = \
    '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/data/Day3_2and3monthOld_rotarodAnalysis/YAC128'  # TODO: User input
userinput_maxFrames = 6000

# dir0 = ({'dir': userinput_0Day3Dir, 'label': str(0)})
# dir1 = ({'dir': userinput_1Day3Dir, 'label': str(1)})
# dirs = [dir0, dir1]

userinput_pBoudn = 0.9
userinput_column_likelihood0 = {'column': ['Rightpaw x', 'Rightpaw y'],
                                 'likelihood': 'Rightpaw likelihood'}

userinput_column_likelihood1 = {'column': ['Leftpaw x', 'Leftpaw y'],
                                 'likelihood': 'Leftpaw likelihood'}

userinput_column_likelihood2 = {'column': ['Tailbase x', 'Tailbase y'],
                                 'likelihood': 'Tailbase likelihood'}

userinput_column_likelihood3 = {'column': ['Rotarodtop x', 'Rotarodtop y'],
                                 'likelihood': 'Rotarodtop likelihood'}

userinput_column_likelihood4 = {'column': ['Rotarodbottom x', 'Rotarodbottom y'],
                                 'likelihood': 'Rotarodbottom likelihood'}

userinput_columns_likelihoods = [userinput_column_likelihood0, userinput_column_likelihood1,
                                 userinput_column_likelihood2, userinput_column_likelihood3,
                                 userinput_column_likelihood4]

output  = fillnan(userinput_columns_likelihoods, userinput_pBoudn,inputDir)
