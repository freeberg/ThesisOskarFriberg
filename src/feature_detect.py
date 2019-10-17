import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from srad import srad

magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (155, 187, "FP2",(100,500), (200, 600))
patients = [FP2, roger, magnus]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_dicom_image(patient, imNbr):
    '''in:patient triple, in:imNbr int'''
    if patient == magnus or patient == FP1:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_00'+str(imNbr)
    else:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_0'+str(imNbr)
    ds = dicom.dcmread(im_path)
    rows = ds.Rows
    cols = ds.Columns
    im1 = np.zeros((rows, cols), np.float32)
    im1[:][:] = rgb2gray(ds.pixel_array[0][:][:])
    f_height = np.array(range(patient[3][0], patient[3][1]))
    f_width = np.array(range(patient[4][0], patient[4][1]))
    focused_img = np.array(im1[:][f_height[:, None],f_width], dtype=np.uint8)
    return focused_img, len(f_height), len(f_width)

def expand_to_c3(img):
    rows, cols = img.shape
    img_c3 = np.ones((rows,cols,3))
    for i in range(rows):
        for j in range(cols):
            img_c3[i][j][:] = 255 * np.log(img[i][j]) * img_c3[i][j][:]

    return img_c3 


sav_or_show = "show"
feat_dic = {}

for patient in patients:
    out_dir = "dataset/" + patient[2] + "_SRAD/"
    print("Producing SRAD for " + patient[2])
    for i in range(patient[1]-patient[0]):
        if patient == FP2 and i == 154:
            continue
        image, rows, cols = load_dicom_image(patient, i + patient[0])
        if('h' in sav_or_show):
            # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
            cl1 = cv2.equalizeHist(image)
            cl2 = clahe2.apply(image)
            scal_img = np.exp(image/255)
            scal_img2 = np.exp(cl2/255)
            srad_img = srad(scal_img, rows, cols, 60, 0.1)
            srad_img2 = srad(scal_img2, rows, cols, 60, 0.1)
            srad_img_c3 = expand_to_c3(srad_img)
            srad_img2_c3 = expand_to_c3(srad_img2)
            cv2.imwrite(out_dir + patient[2] + str(i + patient[0]) + 'ORG.png', srad_img_c3)
            cv2.imwrite(out_dir + patient[2] + str(i + patient[0]) + 'CLAHE.png',srad_img2_c3)
