import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from srad import srad

tobias = (109, 168)
magnus = (42, 69)

imNrs = [str(i) for i in range(42, 69)]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def speckle_reduce(img):
    return img

sav_or_show = "show"
feat_dic = {}

for i in range(int(imNrs[-1])-int(imNrs[0])):
    image = cv2.imread('/home/friberg/Programming/ThesisOskarFriberg/dataset/magnusExt/og/im' + imNrs[i] + 'Original Image.png', 0)

    if('h' in sav_or_show):
        # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        # clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        # cl1 = clahe.apply(image)
        # cl2 = clahe.apply(image)
        rows, cols = image.shape
        scal_img = np.exp(image/255)
        srad_img = srad(scal_img, rows, cols, 20, 0.1)

        # plt.subplot(2,1,1),plt.imshow(cl1 ,'gray')
        # plt.subplot(2,1,2),plt.imshow(cl2 ,'gray')
        plt.subplot(2,1,1),plt.imshow(image ,'gray')
        plt.subplot(2,1,2),plt.imshow(srad_img ,'gray')
        plt.show()
        wait = input("Press enter to continue")
    #else:
        #feat_dic[imNrs[i]] = 


    # f_height = np.array(range(imageCoor[0], imageCoor[1]))
    # f_width = np.array(range(imageCoor[2], imageCoor[3]))
    # corp_img = np.array(image[f_height[:, None],f_width], dtype=np.uint8)

if('a' in sav_or_show):
    w = csv.writer(open("seg_training_data.csv", "w"))
    for key, val in feat_dic.items():
        w.writerow([key, val])   