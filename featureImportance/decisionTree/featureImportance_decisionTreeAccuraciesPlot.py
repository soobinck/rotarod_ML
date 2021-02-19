import os
import string
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from utils.getDirAbsPath import outputAbsPath
from visualization.oneFeaturePlot import fancyBoxPlot

inputCSV = './output.csv'

# Plot definitions

interval = len(string.ascii_lowercase)
df = pd.read_csv(inputCSV, index_col=0)

accuracies = df['accuracy'].values
bins = np.array_split(accuracies, interval)
df = pd.DataFrame(bins)
footnote = 'Testing accuracy of decision tree classification models of WT and YAC128. Training data is the mean step up height in each interval of 30 seconds. Data along the x-axis are cumulative. For example, when x = 0, the tree is only trained with the first interval (30 seconds) of the input data. When x=4, then the model is trained with the data of first 5 intervals.'
# fancyBoxPlot(pd.DataFrame(bins),
#              xlabel='trained up to (i)th interval',
#              ylabel='accuracy',
#              title='Training Accuracy with Different Windows of Time Series Data (step up height) %i iterations' % interval,
#              outputPath=os.path.join(outputAbsPath(), 'featureImportance', 'accuracies.png'),
#              footnote=footnote)

xlabel = 'trained up to (i)th interval'
ylabel = 'accuracy'
title = 'Training Accuracy with Different Windows of Time Series Data (step up height) %i iterations' % interval
outputPath = os.path.join(outputAbsPath(), 'featureImportance', 'accuracies.png')

# Plotting

fig, ax = plt.subplots(figsize=(15, 11), tight_layout=True, constrained_layout=False)
plt.subplots_adjust(hspace=1.0, wspace=0.02, bottom=0.17)

# Creating axes instance
bp = ax.boxplot(df, patch_artist=True,
                notch='True')

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linestyle=":")
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
ft = plt.figtext(0, 0, footnote, wrap=True, va='bottom', fontsize=11)
# ax.annotate(footnote, (0,-0.2), xycoords='figure fraction')
plt.tight_layout()
fig.savefig(outputPath)
# pickle.dump(fig, open((outputPath + '.fig.pickle'), 'wb'))

# show plot
fig.show()
print('%s saved.' % outputPath)

# Plot Importance
