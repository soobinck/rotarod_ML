from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from pytransform3d.rotations import *
from varname import nameof

import sys
import os

sys.path.append("..")
import cv2
from os import listdir
from os.path import isfile, join

# matplotlib.use('tkagg')
# matplotlib.use('WebAgg')


import matplotlib.pyplot as plt
import pandas as pd
import phenograph
import io


def getLastDirectory(inputDir):
    if inputDir.endswith('/'):
        inputDir = inputDir[-1]
    return os.path.split(inputDir)[-1]

day3WT = '/alder/home/soobink/rotarod_ML10/output/Day3_WT'
day3YAC = '/alder/home/soobink/rotarod_ML10/output/Day3_YAC'
day4WT = '/alder/home/soobink/rotarod_ML10/output/Day4_WT'
day4YAC = '/alder/home/soobink/rotarod_ML10/output/Day4_YAC'
day3and4WT = '/alder/home/soobink/rotarod_ML10/output/Day3and4_WT'
day3and4YAC = '/alder/home/soobink/rotarod_ML10/output/Day3and4_YAC'

# paths = [day3WT, day4WT, day3YAC, day4YAC, day3and4WT, day3and4YAC]
paths = [day3and4YAC]
perplexities = [20, 30, 100]

for perplexity in perplexities:
    for path in paths:
        print('Running %s with perplexity = %i.' % (path, perplexity))
        data_2d = [f for f in listdir(path) if (isfile(join(path, f)) and (not f.startswith('.')))]

        # data_3d = ['LD1_1580415036_3d.csv']
        coords_all_2d = []
        coords_all_3d = []
        dataset_name_2d = []
        dataset_name_3d = []

        # for f_2d, f_3d in zip(data_2d, data_3d):
        for f_2d in data_2d:
            coords_file = os.path.join(path, f_2d)
            dataset_name_2d = coords_file
            # coords_2d = pd.read_csv(coords_file, dtype=np.float, header=2, index_col=0)
            coords_2d = pd.read_csv(coords_file, dtype=float, header=0, index_col=0)
            coords_2d.dropna(axis=0, inplace=True)
            coords_2d = coords_2d.values[:, 3:]  # exclude first column
            coords_2d = np.delete(coords_2d, list(range(2, coords_2d.shape[1], 3)),
                                  axis=1)  # delete every 3rd column of prediction score
            coords_all_2d.append(coords_2d)

            # coords_file = data_root + os.sep + f_3d
            # dataset_name_3d = coords_file.split('/')[-1].split('.')[0]
            # coords_3d = pd.read_csv(coords_file, header=2)
            # coords_3d = coords_3d.values[:, 1:]  # exclude the index column
            # coords_3d = np.around(coords_3d.astype('float'), 2)  # round to two decimal places
            # coords_3d = gaussian_filter1d(coords_3d, 5, axis=0)  # smooth the data, the points were oscillating
            # coords_all_3d.append(coords_3d)

        coords_all_2d = np.vstack(coords_all_2d)  # convert to numpy stacked array
        # coords_all_3d = np.vstack(coords_all_3d)
        # x_3d = coords_all_3d[:, ::3];
        # y_3d = coords_all_3d[:, 1::3];
        # z_3d = coords_all_3d[:, 2::3];
        x_2d = coords_all_2d[:, ::2];
        y_2d = coords_all_2d[:, 1::2];
        z_2d = np.zeros(x_2d.shape);
        coords_all_3d_trans = []
        # for i in np.arange(x_3d.shape[0]):

        k = 30  # K for k-means step of phenograph
        communities_2d, graph, Q = phenograph.cluster(coords_all_2d, k=k)
        n_clus_2d = np.unique(communities_2d).shape[0]

        # --end of phenograph

        # tsne_model = TSNE(n_components=2, random_state=2,perplexity=100,angle=0.1,init='pca',n_jobs= mp.cpu_count()-1)
        tsne_model = TSNE(n_components=2, random_state=2, perplexity=perplexity, angle=0.1, init='pca', n_jobs=-1)
        Y_2d = tsne_model.fit_transform(coords_all_2d)
        cmap = plt.cm.colors.ListedColormap(plt.cm.jet(np.linspace(0, 1, n_clus_2d)))
        plt.figure()
        plt.scatter(Y_2d[:, 0], Y_2d[:, 1],
                    c=communities_2d,
                    cmap=cmap,
                    alpha=1.0)
        plt.colorbar(ticks=np.unique(communities_2d), label='Cluster#')
        plt.xlabel('TSNE1');
        plt.ylabel('TSNE2')

        name = getLastDirectory(path)
        plt.title(' 2D Body coordinate clusters: total frames %s\n%s, perplexity = %i' % (
        str(len(communities_2d)), name, perplexity))

        plt.savefig(os.path.join('plots', name + 'p' + str(perplexity) + '.png'), format='png')

        plt.show()
