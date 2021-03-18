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

columnNames = ['rel RightY mm', 'rel LeftY mm', 'rel LeftX mm', 'rel RightX mm', 'Rightpaw euclidean velocity',
               'Leftpaw euclidean velocity', 'wait time b4 step up']

paths = [day3WT, day4WT, day3YAC, day4YAC, day3and4WT, day3and4YAC]
perplexities = [30, 100]
ks = [30, 50, 100, 10]  # K for k-means step of phenograph
for perplexity in perplexities[0]:
    for k in ks[0]:
        for path in paths[0]:
            print('Running %s with k = %i, perplexity = %i.' % (path, k, perplexity))
            data_2d = [f for f in listdir(path) if (isfile(join(path, f)) and (not f.startswith('.')))]

            df = pd.DataFrame(columns=columnNames)
            coords_all_2d = []
            dataset_name_2d = []
            # for f_2d, f_3d in zip(data_2d, data_3d):
            for f_2d in data_2d:
                coords_file = os.path.join(path, f_2d)
                dataset_name_2d = coords_file
                coords_2d = pd.read_csv(coords_file, dtype=float, header=0, index_col=0)
                df_2d = pd.read_csv(coords_file, dtype=float, index_col=0)
                coords_2d.dropna(axis=0, inplace=True)
                coords_2d = coords_2d[columnNames]
                coords_2d = coords_2d.values[:, 3:]  # exclude first column
                coords_2d = np.delete(coords_2d, list(range(2, coords_2d.shape[1], 3)),
                                      axis=1)  # delete every 3rd column of prediction score
                coords_all_2d.append(coords_2d)
                df.append(df_2d)

            coords_all_2d = np.vstack(coords_all_2d)  # convert to numpy stacked array

            x_2d = coords_all_2d[:, ::2];
            y_2d = coords_all_2d[:, 1::2];
            z_2d = np.zeros(x_2d.shape);

            # communities_2d, graph, Q = phenograph.cluster(coords_all_2d, k=k)

            break
