import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset/DICOM 3D/Magnus/IM_0002')

rows = ds.Rows
cols = ds.Columns
pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
nbrOfFrames = ds.NumberOfFrames
frameRate = 1000/ds.FrameTime
threeD = np.zeros((int(nbrOfFrames), rows, cols))
im1 = ds.pixel_array[1]

for i in range(nbrOfFrames):
    threeD[i][:][:] = rgb2gray(ds.pixel_array[i][:][:])
s = "thres"
for i in range(nbrOfFrames):
    f_height = np.array(range(175,435))
    f_width = np.array(range(95,695))
    img = np.array(threeD[i][f_height[:, None],f_width], dtype=np.uint8)

    if 'hi' in s:
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
        plt.imshow(res)
        plt.show()
        wait = input("Press enter to continue")
    else:
        ret,thresh1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,80,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,80,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,80,255,cv2.THRESH_TOZERO_INV)
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()