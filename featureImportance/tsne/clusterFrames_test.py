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

colNames = ['rel RightY mm', 'rel LeftY mm', 'rel LeftX mm', 'rel RightX mm', 'Rightpaw euclidean velocity',
            'Leftpaw euclidean velocity', 'wait time b4 step up']

xCols = ['rel RightY mm', 'rel LeftY mm', 'rel LeftX mm', 'rel RightX mm', 'Rightpaw euclidean velocity',
         'Leftpaw euclidean velocity', 'wait time b4 step up']

perplexity = 30
k = 30
print('Running %s with k = %i, perplexity = %i.' % (day3WT, k, perplexity))
data_2d = [f for f in listdir(day3WT) if (isfile(join(day3WT, f)) and (not f.startswith('.')))]

df = pd.DataFrame()
coords_all_2d = []
dataset_name_2d = []
# for f_2d, f_3d in zip(data_2d, data_3d):
for f_2d in data_2d:
    coords_file = os.path.join(day3WT, f_2d)

    df_2d = pd.read_csv(coords_file, dtype=float, index_col=0)[colNames]

    newColNames = [getLastDirectory(coords_file) + colName for colName in [df_2d.columns]]

    df_file = pd.read_csv(os.path.join
                          (day3WT,
                           f_2d))[colNames].values
    df_file = pd.DataFrame(df_file, columns=newColNames)
    for newColName in newColNames:
        df[newColName] = df_file[newColName]
    dataset_name_2d = coords_file
    coords_2d = pd.read_csv(coords_file, dtype=float, header=0, index_col=0)[colNames]
    coords_2d.dropna()
    coords_2d.dropna(inplace=True)
    coords_all_2d.append(coords_2d)

coords_all_2d = np.vstack(coords_all_2d)  # convert to numpy stacked array

x_2d = coords_all_2d[:, ::2]
y_2d = coords_all_2d[:, 1::2]
z_2d = np.zeros(x_2d.shape)

dummyCol = np.linspace(0, 122350)

# communities_2d, graph, Q = phenograph.cluster(df.values.dropna(axis=0, inplace=False), k=k)
