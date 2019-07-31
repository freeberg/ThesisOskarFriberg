import csv
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dice_loss import dice_coeff
from find_circle import find_circle

man_seg_dict = {}
with open("src/manuellSeg.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        man_seg_dict[row[0]] = [float(row[1]), float(row[2]), float(row[3])]

print(man_seg_dict["Magnus68"])


def evaluate_masks(input_dir, postproc=True):

    # använd manuellSeg.csv, ta ut en dict
    # THEN! Ta ut cirklar från input_masks (från input_path) 
    # Implementera en Dice funktion för cirklar (borde inte vara så svårt)
    # Return dice coef för alla masks och använd som kriteri istället för white space!!

    in_masks = [f for f in os.listdir(input_dir)]
    dice_values = {"Magnus" : 0, "Roger" : 0, "FP1" : 0, "FP2" : 0} #Save the dice coef for every patient!
    tot_dice_coef = 0
    tot_imgs_patients = get_tot_imgs_per_patient(in_masks)
    for i_mask in in_masks:
        if postproc:
            i_c = find_circle(input_dir + i_mask)
            true_c = man_seg_dict[i_mask[0:-len("_seg.png")]]
            dice = dice_coeff_circles(i_c, true_c)
            tot_dice_coef += dice
            if "Magnus" in i_mask or "FP1" in i_mask:
                key = str(i_mask[0:-(len("seg.png") + 3)])
                dice_values[key] += dice / tot_imgs_patients[key]
            else:
                key = str(i_mask[0:-(len("seg.png") + 4)])
                dice_values[key] += dice / tot_imgs_patients[key]
        else:
            print("No post-processing is TBI!")
    return dice_values


def get_tot_imgs_per_patient(in_masks):
    d = {"Magnus" : 0, "Roger" : 0, "FP1" : 0, "FP2" : 0}
    for i_mask in in_masks:
        if "Magnus" in i_mask or "FP1" in i_mask:
            key = str(i_mask[0:-(len("seg.png") + 3)])
            d[key] += 1
        else:
            key = str(i_mask[0:-(len("seg.png") + 4)])
            d[key] += 1
    return d


def dice_coeff_circles(input_circ, target_circ):
    # 2 * |X N Y| / |X| + |Y|
    x_com_y = common_area(input_circ, target_circ)
    x = input_circ[2]**2 * math.pi
    y = input_circ[2]**2 * math.pi
    return 2 * x_com_y / (x + y)


def common_area(A, B):
    print(A)
    d = math.hypot(B[0] - A[0], B[1] - A[1])

    if d < (A[2] + B[2]):
        a = A[2]**2
        b = B[2]**2

        x = (a - b + d * d) / (2 * d)
        z = x * x
        y = math.sqrt(a - z)

        if (d < abs(B[2] - A[2])):
            return math.pi * min(a, b)
        
        return a * math.asin(y / A[2]) + b * math.asin(y / B[2]) - y * (x + math.sqrt(z + b - a))
    return 0

print(evaluate_masks("/home/friberg/Programming/ThesisOskarFriberg/testing_folder/model_sc1_lr0.05_batch1_SGD/sc1_thresh0.2/"))