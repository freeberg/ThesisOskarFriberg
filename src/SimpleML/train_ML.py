import csv
from math import ceil, pi, sin, cos

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure, io
from skimage.feature import hog

from extract_features import extract_HOG, ext_feats_from_data, get_points_in_circle
import operator


def train_ML(patient):
    train_data = "seg_training_data_%s.csv" % patient[2]
    train_set = []
    with open(train_data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            data=[float(i) for i in row[1][1:-1].split(", ")]
            train_set.append((row[0], data))
    feat_dict = ext_feats_from_data(train_set, patient, 10)
    class_data = get_class_data_from_feat(feat_dict)
    return class_data


def get_class_data_from_feat(feat_dict):
    class_data = {}
    circles = list(feat_dict.values())
    min_len = min(circles, key=len)
    stds, means, upper_stds, upper_means, lower_stds, lower_means = ([],[],[],[],[],[])
    for i in range(len(circles)):
        means.append(np.mean(circles[i]))
        upper_half = get_half_circle(circles[i], 'u')
        lower_half = get_half_circle(circles[i], 'l')
        upper_stds.append(np.std(upper_half))
        upper_means.append(np.mean(upper_half))
        lower_stds.append(np.std(lower_half))
        lower_means.append(np.mean(lower_half))
        stds.append(np.std(circles[i]))

    class_data["std"] = stds
    class_data["upper_std"] = upper_stds
    class_data["lower_std"] = lower_stds
    class_data["mean"] = means
    class_data["upper_mean"] = upper_means
    class_data["lower_mean"] = lower_means


    return class_data


# def circle_classes(circle):
#     upper_half = get_half_circle(circle, 'u')
#     lower_half = get_half_circle(circle, 'l')
#     class_dict = {"std":np.std(circle), "mean":np.mean(circle),
#                   "upper_std":np.std(upper_half), "upper_mean":np.mean(upper_half),
#                   "lower_std":np.std(lower_half), "lower_mean":np.mean(lower_half)}
#     return class_dict


def get_half_circle(circle, option):
    if 't' in option:
        return circle[:int(len(circle)/2)]
    else:
        return circle[int(len(circle)/2):]






# train_ML("seg_training_data_Magnus.csv", "Magnus")