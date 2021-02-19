import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.getDirAbsPath import getLastDirectory
import seaborn as sns


def stepUp2HeightPlot(dir_labelDictionaries, secondsPerInterval=30):
    dir_labelDictionary0 = dir_labelDictionaries[0]
    df0 = pd.read_csv(dir_labelDictionary0['dir'], index_col=0)
    df0 = df0.transpose()
    xaxis = (df0.columns + 1) * secondsPerInterval
    df0['label'] = dir_labelDictionary0['label'] + ' (n = %i)' % len(df0)

    dir_labelDictionary1 = dir_labelDictionaries[1]
    df1 = pd.read_csv(os.path.join(dir_labelDictionary1['dir']), index_col=0)
    df1 = df1.transpose()
    df1['label'] = dir_labelDictionary1['label'] + '(n = %i)' % len(df1)

    df = pd.concat([df0, df1])
    df = df.melt(id_vars='label')
    ax = sns.boxplot(data=df, x='variable', y='value', hue='label')
    ax.set_xlabel('seconds')
    ax.set_ylabel('Step Up Height(mm)')
    ax.set_xticklabels(xaxis)
    plt.title('Step Up Height of Right Foot')
    plt.legend()
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # outputDir = os.path.split(inputAbsolutePath)[0]
    # outputFile = getLastDirectory(inputAbsolutePath)[:-4] + '.png'
    # fig.savefig(os.path.join(outputDir, outputFile))
    # pickle.dump(fig, open(os.path.join(outputDir, outputFile + '.fig.pickle'), 'wb'))

    # show plot
    plt.show()
    # print('%s saved at %s' % (outputFile, outputDir))
