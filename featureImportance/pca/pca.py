from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from pytransform3d.rotations import *
from varname import nameof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append("..")
import cv2
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import pandas as pd
import phenograph
import io


def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='green', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


def getLastDirectory(inputDir):
    if inputDir.endswith('/'):
        inputDir = inputDir[-1]
    return os.path.split(inputDir)[-1]


day3WT = os.path.join('..', '..', 'output', 'Day3_WT')
day3YAC = os.path.join('..', '..', 'output', 'Day3_YAC')
day4WT = os.path.join('..', '..', 'output', 'Day4_WT')
day4YAC = os.path.join('..', '..', 'output', 'Day4_YAC')
day3and4WT = os.path.join('..', '..', 'output', 'Day3and4_WT')
day3and4YAC = os.path.join('..', '..', 'output', 'Day3and4_YAC')

paths = [day3WT, day4WT, day3YAC, day4YAC, day3and4WT, day3and4YAC]
# paths = [day3and4YAC]
columnNames = ['rel RightY mm', 'rel LeftY mm', 'rel LeftX mm', 'rel RightX mm', 'Rightpaw euclidean velocity',
               'Leftpaw euclidean velocity', 'wait time b4 step up']
# for path in paths:
for path in paths:
    data_2d = [f for f in listdir(path) if (isfile(join(path, f)) and (not f.startswith('.')))]

    # data_3d = ['LD1_1580415036_3d.csv']
    x = pd.DataFrame()
    coords_all_3d = []
    dataset_name_3d = []

    # for f_2d, f_3d in zip(data_2d, data_3d):
    for f_2d in data_2d:
        coords_file = os.path.join(path, f_2d)
        # coords_2d = pd.read_csv(coords_file, dtype=np.float, header=2, index_col=0)
        coords_2d = pd.read_csv(coords_file, dtype=float, header=0, index_col=0)
        coords_2d.dropna(axis=0, inplace=True)
        coords_2d = coords_2d[columnNames]

    x = x.append(coords_2d)
    x = x.values
    stdScaler = StandardScaler()
    x = stdScaler.fit_transform(x)
    x = pd.DataFrame(x, columns=columnNames)

    name = getLastDirectory(path)

    pcamodel = PCA(n_components=5)
    pca = pcamodel.fit_transform(x)

    figs, axs = plt.subplots(3, figsize=(18, 13))
    figs.suptitle('PCA Analysis - %s' % name)
    axs[0].bar(range(1, len(pcamodel.explained_variance_) + 1), pcamodel.explained_variance_)
    axs[0].set(xlabel='Components', ylabel='Explained variance')

    axs[0].plot(range(1, len(pcamodel.explained_variance_) + 1),
                np.cumsum(pcamodel.explained_variance_),
                c='red',
                label="Cumulative Explained Variance")

    axs[0].legend(loc='upper left')

    axs[1].plot(pcamodel.explained_variance_ratio_)
    axs[1].set(xlabel='number of components', ylabel='cumulative explained variance')

    axs[2].scatter(pca[:, 0], pca[:, 1])
    axs[2].set(xlabel='PCA1', ylabel='PCA2')

    axs[2].set_title('PCA: total frames %s - %s' % (
        str(len(x)), name))

    myplot(pca[:, 0:2], np.transpose(pcamodel.components_[0:2, :]), list(x.columns))
    plt.show()

    figs.tight_layout()
    figs.savefig(os.path.join('.', 'plots', name + '.png'), format='png')
    plt.show()

    heatmap, ax = plt.subplots(1)
    ax = sns.heatmap(pcamodel.components_,
                     cmap='YlGnBu',
                     yticklabels=["PCA" + str(x) for x in range(1, pcamodel.n_components_ + 1)],
                     xticklabels=list(x.columns),
                     cbar_kws={"orientation": "vertical"})
    heatmap.autofmt_xdate()

    ax.set_aspect("equal")

    heatmap.savefig(os.path.join('plots', 'heatmap_' + name  + '.png'), format='png')
