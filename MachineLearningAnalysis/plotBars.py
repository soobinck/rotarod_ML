import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

def plotML():
    pass

inputDir = './output'
inputCSV = 'accuracies.csv'
df = pd.read_csv(os.path.join(inputDir, inputCSV), index_col=0)
# color = dict(boxes='blue', whiskers='blue', medians='red', caps='blue')
# styles=dict(linewidth=20)
# df.plot.box(figsize=(40,20), fontsize=30, color=color, style=styles)
#
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(df, patch_artist=True,
                notch='True')

colors = [] # ['#0000FF']

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
ax.set_xticklabels(df.columns)
ax.set_xlabel('model')
ax.set_ylabel('accuracy')
# Adding title
plt.title('Test Accuracy of 26 Trials \n len(X) = 133, test split = 0.3')

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.text(1,0,
        'template from https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/',
        verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
        fontsize=7
        )
fig.savefig(os.path.join('./output', inputCSV[:-4] + '.png'))
pickle.dump(fig, open(os.path.join('./output', inputCSV[:-4] + '.fig.pickle'), 'wb'))


# show plot
plt.show()
