import os
import shutil

import pandas as pd

# sampleFileName = 'Experiment2-190630_Day4_145m1_rotarod3_Cam2onRotarodDeepCut_resnet50_rotarod3Jul17shuffle1_1030000'

toDelete = ['Experiment2-', 'Cam2onRotarod', 'Deepcut', 'DeepCut', 'convertedDeepCut_', 'resnet50_', 'rotarodpaws_',
            'shuffle1_1030000', 'Trimmed_before_rotation', 'Trimmed_at_end', '-converted_', '_rotated_', '_Trimmed_',
            '_Trimmed_at_End',
            'relabeled_combined_']


def deleteCommonWordsInFileName(fileName):
    for item in toDelete:
        fileName = fileName.replace(item, '')
    fileName = fileName.replace('__', '_')
    return fileName

