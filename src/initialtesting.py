import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

startIm = 109
imageCoor = [120, 510, 160, 675]

for i in range(50):
    ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset/Tobias/IM_0' + str(startIm))

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
    f_height = np.array(range(imageCoor[0], imageCoor[1]))
    f_width = np.array(range(imageCoor[2], imageCoor[3]))
    img = np.array(threeD[i][f_height[:, None],f_width], dtype=np.uint8)

    if 'hi' in s:
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side FCN U-net
        plt.imshow(res)
        plt.show()
        wait = input("Press enter to continue")
    else:
        ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_TRUNC)
        ret,thresh2 = cv2.threshold(img,50,255,cv2.THRESH_TOZERO)
        titles = ['Original Image','TRUNC','TOZERO']
        images = [img, thresh1, thresh2]
        for i in range(3):
            # plt.subplot(3,1,i+1),plt.imshow(images[i],'gray')
            # plt.title(titles[i])
            # plt.xticks([]),plt.yticks([])
            cv2.imwrite("im"+str(startIm)+titles[i]+".png", images[i])

        plt.show()
    startIm = startIm + 1