import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset//Tobias/IM_0109')

rows = ds.Rows
cols = ds.Columns
pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
nbrOfFrames = ds.NumberOfFrames
frameRate = 1000/ds.FrameTime
threeD = np.zeros((int(nbrOfFrames), rows, cols))
im1 = ds.pixel_array[1]
print(ds)

for i in range(nbrOfFrames):
    threeD[i][:][:] = rgb2gray(ds.pixel_array[i][:][:])


# arg = input("Whole (w) or focused (f)?")

for j in range(nbrOfFrames):
    if False:
        im = threeD[j][:][:]
        plt.imshow(im)
        plt.show()
        wait = input("Press enter to continue")
    else:
        f_height = np.array(range(175,435))
        f_width = np.array(range(95,695))
        print(threeD.shape)
        im = threeD[j+40][:][f_height[:, None],f_width]
        plt.imshow(im)
        plt.show()
        wait = input("Press enter to continue")
        cv2.imwrite('im1.png',im)
