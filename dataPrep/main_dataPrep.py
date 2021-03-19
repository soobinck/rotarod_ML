import shutil

from dataPrep import *
import pandas as pd
import os
import pathlib
from distutils.dir_util import copy_tree
from utils.getDirAbsPath import outputAbsPath


def prepareData(dir_labelDictionary):
    userinput_pBoudn = 0.9
    userinput_column_likelihood0 = {'column': ['Rightpaw x', 'Rightpaw y'],
                                    'likelihood': 'Rightpaw likelihood'}

    userinput_column_likelihood1 = {'column': ['Leftpaw x', 'Leftpaw y'],
                                    'likelihood': 'Leftpaw likelihood'}

    userinput_column_likelihood2 = {'column': ['Tailbase x', 'Tailbase y'],
                                    'likelihood': 'Tailbase likelihood'}

    userinput_column_likelihood3 = {'column': ['Rotarodtop x', 'Rotarodtop y'],
                                    'likelihood': 'Rotarodtop likelihood'}

    userinput_column_likelihood4 = {'column': ['Rotarodbottom x', 'Rotarodbottom y'],
                                    'likelihood': 'Rotarodbottom likelihood'}

    userinput_columns_likelihoods = [userinput_column_likelihood0, userinput_column_likelihood1,
                                     userinput_column_likelihood2, userinput_column_likelihood3,
                                     userinput_column_likelihood4]

    outputDirAbsolutePath = outputAbsPath()

    for dir_label in dir_labelDictionary:
        inputDir = dir_label['dir']
        label = dir_label['label']

        outputDir = os.path.join(outputDirAbsolutePath, label)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        # TODO: For each functions below, add progress bar, especially fillnan (t).
        lastDir = \
            idxCSVs(
                isStepUpFrame(
                    addVelocityColumnsBothFeet(
                        pixel2mm(
                            fillnan(userinput_columns_likelihoods, userinput_pBoudn,
                                    cleanCSV(inputDir))))))

        copy_tree(lastDir, outputDir)
        shutil.rmtree(lastDir)
    return outputDir

