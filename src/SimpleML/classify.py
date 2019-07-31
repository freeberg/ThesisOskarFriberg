from collections import defaultdict
from math import cos, pi, sin, sqrt, ceil, log

import cv2
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from train_ML import train_ML, get_half_circle
from extract_features import extract_HOG, ext_feats_from_data, get_img_path

magnus = (42, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 169, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
train_patient = magnus
patient = tobias

def find_best_circle(patient, train_patient, img_nr, r, padding=10):

    class_data = train_ML(train_patient)
    # r = r + r/padding
    pad = r/padding
    img = cv2.imread(get_img_path(img_nr, patient[2]), cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    circle_pos = [(x, y) for x in range(ceil(r + pad),int(rows - (r + pad)) - 2) for y in range(ceil(r + pad),int(cols - (r+pad)) - 2)]
    
    circle_scores = {}
    for (x, y) in circle_pos:
        feat_dict = ext_feats_from_data([[img_nr,(x, y, r)]], patient, 10)
        circle_scores[(x,y)] = eval_circle(feat_dict[0], class_data)
    
    best_center = (0,0)
    best_score = -10000
    for k in circle_scores:
        score = 0
        for v in circle_scores[k].values():
            score += v
        if score > best_score:
            best_score = score
            print(k)
            best_center = k
    
    return best_center




def eval_circle(circle, train_data):
    upper_half = get_half_circle(circle, 'u')
    lower_half = get_half_circle(circle, 'l')
    circ_classes = {"std":np.std(circle), "mean":np.mean(circle),
                    "upper_std":np.std(upper_half), "upper_mean":np.mean(upper_half),
                    "lower_std":np.std(lower_half), "lower_mean":np.mean(lower_half)}
    logprob_class = {}
    for k in train_data.keys():
        logprob_class[k] = log(bayes(train_data[k], circ_classes[k]))
        # print(k, logprob_class)
    return logprob_class


def bayes(train_class, circ_class):
    mean_of_classes = np.mean(train_class)
    std_of_classes = np.std(train_class)
    return norm(mean_of_classes, std_of_classes).pdf(circ_class)

# [m n] = size(x);
# nr_classes = size(classification_data, 2) / 3;

# mean_features = classification_data(:, 1:nr_classes);
# std_features = classification_data(:, (nr_classes + 1):(nr_classes * 2));
# p_classes = classification_data(1,(nr_classes * 2 + 1):(nr_classes * 3));

# p_ys = zeros(nr_classes, 1);

# for i = 1:m
#     for k = 1:nr_classes
#         p_ys(k) = p_ys(k) + log(normpdf(x(i,1), mean_features(i,k), std_features(i, k)));
#     end
# end

# for j = 1:nr_classes
#     p_ys(j) = log(p_classes(j)) + p_ys(j);
# end

# best_p = max(p_ys);
# y = find(p_ys==best_p, 1)



print(find_best_circle(patient, train_patient, "223", 42.134823299018805))


