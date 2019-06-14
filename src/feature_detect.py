import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from srad import srad

magnus = (69, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 169, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
patient = magnus

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_dicom_image(patient, imNbr):
    '''in:patient triple, in:imNbr int'''
    im_path = '/home/friberg/Programming/ThesisOskarFriberg/dataset/'+patient[2]+'/IM_00'+str(imNbr)
    ds = dicom.dcmread(im_path)
    rows = ds.Rows
    cols = ds.Columns
    im1 = np.zeros((rows, cols), np.float32)
    im1[:][:] = rgb2gray(ds.pixel_array[0][:][:])
    f_height = np.array(range(patient[3][0], patient[3][1]))
    f_width = np.array(range(patient[4][0], patient[4][1]))
    focused_img = np.array(im1[:][f_height[:, None],f_width], dtype=np.uint8)
    return focused_img, len(f_height), len(f_width)


sav_or_show = "show"
feat_dic = {}

for i in range(patient[1]-patient[0]):
    image, rows, cols = load_dicom_image(patient, i + patient[0])
    if('h' in sav_or_show):
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        # cl1 = cv2.equalizeHist(image)
        cl2 = clahe2.apply(image)
        scal_img = np.exp(image/255)
        scal_img2 = np.exp(cl2/255)
        srad_img = srad(scal_img, rows, cols, 120, 0.05)
        srad_img2 = srad(scal_img2, rows, cols, 120, 0.05)
        cv2.imwrite(patient[2] + str(i + patient[0]) + 'og2.png',255 * np.log(srad_img))
        cv2.imwrite(patient[2] + str(i + patient[0]) + 'he2.png',255 * np.log(srad_img2))
        # wait = input("Press enter to continue")
    #else:
        #feat_dic[imNrs[i]] = 


    # f_height = np.array(range(imageCoor[0], imageCoor[1]))
    # f_width = np.array(range(imageCoor[2], imageCoor[3]))
    # corp_img = np.array(image[f_height[:, None],f_width], dtype=np.uint8)

# if('a' in sav_or_show):
#     w = csv.writer(open("seg_training_data.csv", "w"))
#     for key, val in feat_dic.items():
#         w.writerow([key, val])   