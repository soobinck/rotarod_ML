import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.getDirAbsPath import getLastDirectory


def stepUpHeightPlot(inputAbsolutePath, secondsPerInterval=30):
    df = pd.read_csv(os.path.join(inputAbsolutePath), index_col=0)
    df = df.transpose()

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = df.boxplot(patch_artist=True,
                    notch='True',
                    return_type='dict')

    colors = []

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    ax.set_xticklabels((df.columns + 1) * secondsPerInterval)
    ax.set_xlabel('seconds')
    ax.set_ylabel('Step Up Height(mm)')
    # Adding title
    plt.title('Step Up Height of %i experiments (%s)' % (len(df), getLastDirectory(inputAbsolutePath)))

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.text(1, 0,
            'template from https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/',
            verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
            fontsize=7
            )
    outputDir = os.path.split(inputAbsolutePath)[0]
    outputFile = getLastDirectory(inputAbsolutePath)[:-4] + '.png'
    fig.savefig(os.path.join(outputDir, outputFile))
    pickle.dump(fig, open(os.path.join(outputDir, outputFile + '.fig.pickle'), 'wb'))

    # show plot
    plt.show()
    print('%s saved at %s' %(outputFile, outputDir))
