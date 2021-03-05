from sklearn.manifold import TSNE
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

df = pd.DataFrame([1, 2, 3])


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
perplexities = [30, 100]
ks = [30, 50, 100, 10]  # K for k-means step of phenograph
for perplexity in perplexities:
    for k in ks:
        for path in paths:
            print('Running %s with k = %i, perplexity = %i.' % (path, k, perplexity))
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

            communities_2d, graph, Q = phenograph.cluster(coords_all_2d, k=k)
            n_clus_2d = np.unique(communities_2d).shape[0]
