import os
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.getDirAbsPath import outputAbsPath

inputCSV = './output.csv'

# Plot importance when the number of intervals is 2.

interval = len(string.ascii_lowercase)
df = pd.read_csv(inputCSV, index_col=0)
df01 = df[['0', '1']]

bins = np.array_split(df01, len(df01) / interval)[1]
df01 = pd.DataFrame(bins)

footnote = ''
# fancyBoxPlot(pd.DataFrame(bins),
#              xlabel='trained up to (i)th interval',
#              ylabel='accuracy',
#              title='Training Accuracy with Different Windows of Time Series Data (step up height) %i iterations' % interval,
#              outputPath=os.path.join(outputAbsPath(), 'featureImportance', 'accuracies.png'),
#              footnote=footnote)

xlabel = '(i)th interval'
ylabel = 'feature importance'
title = 'Feature importance in classifying genotypes (%i iterations)' % interval
outputPath = os.path.join(outputAbsPath(), 'featureImportance', 'featureImportanceDecisionTree.png')
footnote = 'Classified with mean step up height as the model\'s input. Classified with mean step up height as the model\'s input. Classified with mean step up height as the model\'s input. Classified with mean step up height as the model\'s input. Classified with mean step up height as the model\'s input. '
# Plotting

fig, ax = plt.subplots(figsize=(15, 11), tight_layout=True)
plt.subplots_adjust(hspace=1.0, wspace=0.02, bottom=0.17)

# Creating axes instance
bp = ax.boxplot(bins, patch_artist=True,
                notch='True')

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linestyle="-.", linewidth=3)
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B')

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color='red')

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
fig.subplots_adjust(bottom=0.2)
# ft = plt.figtext(0, 0, footnote, wrap=True, va='bottom', fontsize=11)
# ax.annotate(footnote, (0,-0.2), xycoords='figure fraction')
plt.tight_layout()
plt.grid(True, ls='-.')
fig.savefig(outputPath)
# pickle.dump(fig, open((outputPath + '.fig.pickle'), 'wb'))

# show plot
fig.show()
print('%s saved.' % outputPath)
