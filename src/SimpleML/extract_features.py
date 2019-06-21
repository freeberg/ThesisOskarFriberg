import csv
import operator
from math import ceil, cos, pi, sin

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure, io
from skimage.feature import hog


def extract_HOG(image_path, show=False):
    image = io.imread(image_path)

    fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # If show=True show the extracted features
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
    
    return fd, hog_image

def ext_feats_from_data(data_set, patient, extra_margin, show=False):
    feat_dict = {}
    for i in range(len(data_set)):
        data = data_set[i]
        img = cv2.imread(get_img_path(data[0], patient[2]), 0)
        fd, hog_img = extract_HOG(get_img_path(data[0], patient[2]))
        x_center, y_center, r = data[1]

        circle_indeces = get_points_in_circle(hog_img, ceil(x_center), 
                                ceil(y_center), ceil(r + r/extra_margin))

        feat_dict[i] = (np.hstack(circle_indeces[circle_indeces != 0]))
        if (show):
            im_circle_indeces = get_points_in_circle(img, ceil(x_center), 
                        ceil(y_center), ceil(r + r/extra_margin))
            plt.subplot(2,1,1),plt.imshow(circle_indeces ,'gray')
            plt.subplot(2,1,2),plt.imshow(im_circle_indeces ,'gray')
            plt.show()
        #https://medium.com/the-downlinq/histogram-of-oriented-gradients-hog-heading-classification-a92d1cf5b3cc

    return feat_dict

def get_points_in_circle(hog_img, x_center, y_center, r):
    points = np.zeros((2*r+1, 2*r+1))
    for x in range(x_center - r, x_center + r + 1):
        for y in range(y_center - r, y_center + r + 1):
            if ((x - x_center)**2 + (y - y_center)**2) <= r**2:
                points[x - x_center - r][y - y_center - r] = hog_img[x][y]
            # else:
                # points[x - x_center - r][y - y_center - r] = 0.00001
        
    return points


def get_img_path(im_nbr, patient_name):
    return "../dataset/" + patient_name + "SRAD/" + patient_name + im_nbr + "he2.png"



# extract_HOG("/home/friberg/Programming/ThesisOskarFriberg/dataset/tobSRAD/Tobias120he2.png", True)
