import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.getDirAbsPath import outputAbsPath
from utils.throwError import isCSVFile


def waitTimeB4StepUpPlot(dir_labelDictionaries, secondsPerInterval, n, frames_per_sec, max_est):
    col = 'step up waittime'
    step = frames_per_sec * secondsPerInterval
    frames_per_min = frames_per_sec * 60
    min_per_step = step / frames_per_min

    outputDir = outputAbsPath()
    picklePath = outputAbsPath()

    waittimes_WT = np.zeros([n, 2])
    waittimes_YAC = np.zeros([n, 2])

    for dir_labelDictionary in dir_labelDictionaries:
        inputDir = dir_labelDictionary['dir']
        label = dir_labelDictionary['label']

        for roots, dirs, files in os.walk(inputDir):
            if label == '0':
                waitTime = waittimes_WT
                files_WT = files
            else:
                waitTime = waittimes_YAC
                files_YAC = files

            for inputFile in files:
                if not isCSVFile(inputFile):
                    continue

                df = pd.read_csv(os.path.join(inputDir, inputFile))
                dfs = df[col]
                df_len = len(dfs)
                for i in range(n):
                    start = i * step
                    end = (i + 1) * step
                    if end < len(dfs):
                        splits = dfs[start: end]
                        splits1 = splits != 0
                        waittime = np.sum(splits)
                        n_waittime = np.sum(splits1)
                        waitTime[i][0] += waittime
                        waitTime[i][1] += n_waittime
                    elif start < len(dfs):
                        splits = dfs[i * step:]
                        splits1 = splits != 0
                        waittime = np.sum(splits)
                        n_waittime = np.sum(splits1)
                        waitTime[i][0] += waittime
                        waitTime[i][1] += n_waittime
                    else:
                        break

    # the unit of x is minute
    x = ((pd.Series(range(1, n + 1)) * step) / frames_per_min)
    y_WT = waittimes_WT[:, 0] / waittimes_WT[:, 1] / frames_per_sec
    y_YAC = waittimes_YAC[:, 0] / waittimes_YAC[:, 1] / frames_per_sec
    f, ax = plt.subplots(1, figsize=(40, 20))

    ax.plot(x, y_WT, linewidth=7, label='WT  n=%i' % len(files_WT))
    ax.scatter(x, y_WT, marker='o', s=150)

    ax.plot(x, y_YAC, linewidth=7, label='YAC n=%i' % len(files_YAC))
    ax.scatter(x, y_YAC, marker='o', s=150)
    ax.legend(shadow=True, fontsize=30)

    # xcoords = np.arange(0, max(x), 0.5)
    ax.set_xlim(xmin=0)
    ax.set_xlabel('nth %.1f minute' % min_per_step, fontsize=30)
    ax.set_ylabel('wait time in second', fontsize=30)
    ax.set_title('Mean wait time before a step up', fontsize=30, fontweight='bold', y=1.01)
    ax.tick_params(labelsize=30)

    plt.tight_layout()
    f.savefig(os.path.join(outputDir, 'waittime' + '.png'))
    pickle.dump(f, open(os.path.join(picklePath, 'waittime' + '.fig.pickle'), 'wb'))

    plt.show()
