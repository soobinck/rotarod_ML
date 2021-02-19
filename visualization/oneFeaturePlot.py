import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.getDirAbsPath import getLastDirectory
import matplotlib.gridspec as gridspec


def fancyBoxPlot(df, xlabel, ylabel, title, outputPath, footnote=''):
    fig = plt.figure(figsize=(15,9))
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Adding title
    plt.title(title)


    plt.figtext(1, 0.01, footnote,wrap=True,horizontalalignment='right')
    # ax.annotate(footnote, (0,0), xycoords='figure fraction')
    ax.reset_position()
    plt.tight_layout(pad=0.4)

    fig.savefig(outputPath)
    pickle.dump(fig, open((outputPath + '.fig.pickle'), 'wb'))

    # show plot
    plt.show()
    print('%s saved.' % outputPath)
