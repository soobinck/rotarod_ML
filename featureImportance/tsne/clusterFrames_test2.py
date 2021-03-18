from pytransform3d.rotations import *
from varname import nameof

import sys
import os
from numpy import random

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
            print('Running %s with perplexity = %i.' % (path, perplexity))
            data_2d = [f for f in listdir(path) if (isfile(join(path, f)) and (not f.startswith('.')))]

            coords_all_2d = []
            coords_all_3d = []
            dataset_name_2d = []
            dataset_name_3d = []

            for f_2d in data_2d:
                coords_file = os.path.join(path, f_2d)
                dataset_name_2d = coords_file
                coords_2d = pd.read_csv(coords_file, dtype=float, header=0, index_col=0)

                coords_2d['file path'] = coords_file
                coords_2d['nth frame'] = coords_2d.index + 1
                coords_2d['nth second'] = (coords_2d.index + 1) / 20

                coords_2d.dropna(axis=0,
                                 inplace=True)  # drop rows with na values. Please note that it will only drop the first and last few columns due to the fillnan function in dataCleaning process.

                for col in coords_2d.columns:  # if the column name contains "Unnamed" or "likelihood", which is often genereated by Pandas and deeplabcut.
                    if 'Unnamed' in col:
                        coords_2d = coords_2d.drop(columns=[col])

                    if 'likelihood' in col:
                        coords_2d = coords_2d.drop(columns=[col])

                headers = coords_2d.columns

                # check if the next column is the index column
                if np.array_equal(coords_2d.iloc[1:, 0].values, coords_2d.iloc[:-1, 0].values + 1):
                    print('The first column is the index column.')
                    coords_2d.index = coords_2d.iloc[:, 0]
                    coords_2d = coords_2d.drop(coords_2d.columns[0], axis=1)
                    headers = headers[1:]

                coords_2d = coords_2d.values

                coords_all_2d.append(coords_2d)

            coords_all_2d = np.vstack(coords_all_2d)  # convert to numpy stacked array

            x_2d = coords_all_2d[:, ::2];
            y_2d = coords_all_2d[:, 1::2];
            z_2d = np.zeros(x_2d.shape);

            # communities_2d, graph, Q = phenograph.cluster(coords_all_2d, k=k)

            communities_2d = random.randint(33 + 1, size=coords_all_2d.shape[0])
            df = pd.DataFrame(coords_all_2d, columns=headers)
            df["Clustering"] = communities_2d



            break
