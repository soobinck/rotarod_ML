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
import matplotlib

# matplotlib.use('tkagg')
# matplotlib.use('WebAgg')


import matplotlib.pyplot as plt
import pandas as pd
import phenograph
import io


def dataAbsPath(relativePath=os.path.join('./')):
    absolutePathofRelativePath = os.path.abspath(relativePath).split('/')
    # return os.path.join('/',
    #                     *absolutePathofRelativePath[:
    #                                                 absolutePathofRelativePath.index('rotarod_ML4') + 1]
    #                     , 'data_all')
    return '/tmp/rotarod_ML4/data_all'


# Check if current environment is google colab.
# If so, execute following specific lines

# @title function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


data_root = dataAbsPath()

# %%
# mouse wheel joints
joints = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
          [10, 11], [11, 12], [12, 13], [14, 15], [16, 17], [17, 18], [18, 19], [19, 20],
          [20, 21], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27]]

data_all = dataAbsPath()

day3WT = '/alder/home/soobink/rotarod_ML10/data_all/Day3_2and3monthOld_rotarodAnalysis/WT'
day3YAC = '/alder/home/soobink/rotarod_ML10/data_all/Day3_2and3monthOld_rotarodAnalysis/YAC128'
day4WT = '/alder/home/soobink/rotarod_ML10/data_all/Day4_2and3monthOld_rotarodAnalysis/WT'
day4YAC = '/alder/home/soobink/rotarod_ML10/data_all/Day4_2and3monthOld_rotarodAnalysis/YAC128'

paths = [day3WT, day3YAC, day4WT, day4YAC]

for path in paths:
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
        coords_2d = pd.read_csv(coords_file, dtype=np.float, header=2, index_col=0)
        coords_2d.dropna(axis=0, inplace=True)
        coords_2d = coords_2d.values[:, 1:]  # exclude first column
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
    tsne_model = TSNE(n_components=2, random_state=2, perplexity=100, angle=0.1, init='pca', n_jobs=-1)
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
    plt.title('2D Body coordinate clusters: total frames ' + str(len(communities_2d)))
    plt.savefig('syn_2d_tsne.png', format='png')
    plt.show()
    plt.close()
