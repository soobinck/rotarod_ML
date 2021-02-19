import numpy as np


def calc_means_stds_threshold(dfs_labels_arr, bound):
    n = len(dfs_labels_arr)
    means_labels = np.zeros([n, 3])

    t_bound = 0
    t_all = 0
    for i in range(n):
        df = (dfs_labels_arr[i])[0]
        df.shape[0]
        df_bound = df[df['likelihood'] > bound]
        (means_labels[i])[2] = (dfs_labels_arr[i])[1]
        (means_labels[i])[0] = df_bound['y'].mean()
        (means_labels[i])[1] = df_bound['y'].std()
        t_bound += df_bound.shape[0]
        t_all += df.shape[0]

    print('%.2f%% of frames with likelihood > %.2f' % (t_bound / t_all * 100, bound))

    return means_labels[:, 0:2], means_labels[:, 2]
