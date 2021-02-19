from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from dataAnalysis.distribution.concatCSVs import concatenateCSVs

userinput_0Dir = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/cleanedAnalyzed/0'
userinput_1Dir = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/cleanedAnalyzed/1'

dir0 = ({'dir': userinput_0Dir, 'label': str(0)})
dir1 = ({'dir': userinput_1Dir, 'label': str(1)})
dirs = [dir0, dir1]

[WT, YAC128] = concatenateCSVs(dirs)
# WT = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/WTCombined.csv'
# YAC128 = '/Users/ksb7640/Documents/UBC_Academic/Raymond_Lab/448/rotarod_git/rotarod_ML/output/YACCombined.csv'


# df_WT = pd.read_csv(WT, header=None, names=['WT'])
df_WT = pd.read_csv(WT)
df = pd.DataFrame()

df['WT'] = df_WT['Rightpaw y']

# df_YAC128 = pd.read_csv(YAC128, header=None, names=['YAC128'])
df_YAC128 = pd.read_csv(YAC128)
# df = pd.concat([df_WT, df_YAC128], axis=1)

df['YAC'] = df_YAC128['Rightpaw y']


ax = df.plot.kde(title='Kernel density estimate plot using Gaussian kernels')
fig = ax.get_figure()
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 3)
ax.set_xlabel('Y-coordinate of right foot - scaled')
fig.savefig('KDE_right_y.png')
plt.show()

print(np.mean(df_WT))
print(np.mean(df_YAC128))
