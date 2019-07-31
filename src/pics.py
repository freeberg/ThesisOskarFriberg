import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

magnus = (44, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 169, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1")
FP2 = (154, 187, "FP2")
patient = FP2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# print(ds)

# for i in range(nbrOfFrames):
#     threeD[i][:][:] = rgb2gray(ds.pixel_array[i][:][:])


# arg = input("Whole (w) or focused (f)?")

for j in range(patient[0],patient[1]):
    if patient == FP2 and j == 154:
        continue


    print("image nbr: " + str(j))

    ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset/' + patient[2] + '/IM_0' + str(j))

    rows = ds.Rows
    cols = ds.Columns
    pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    nbrOfFrames = ds.NumberOfFrames
    frameRate = 1000/ds.FrameTime
    im1 = ds.pixel_array[1]
    if False:
        plt.imshow(im1)
        plt.show()
        wait = input("Press enter to continue")
    else:
        f_height = np.array(range(100,500))
        f_width = np.array(range(200,600))
        focused_im = im1[:][f_height[:, None],f_width]
        # plt.imshow(focused_im)
        print(focused_im.shape)
        # plt.show()
        # wait = input("Press enter to continue")
        cv2.imwrite(patient[2] + str(j) + "_c3.png",focused_im)
