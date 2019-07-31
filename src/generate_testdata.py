import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

magnus = (42, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (105, 187, "FP2",(100,500), (200, 600))
patients = [magnus, tobias, roger, FP1, FP2]

def generate_img(nbr, output_dir=""):
    if patient == FP2 and j == 154:
        return

    print("image nbr: " + str(j))
    
    if patient == magnus or patient == FP1:
        ds = dicom.dcmread('dataset/' + patient[2] + '/IM_00' + str(j))
    else:
        ds = dicom.dcmread('dataset/' + patient[2] + '/IM_0' + str(j))
        

    # rows = ds.Rows
    # cols = ds.Columns
    # pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    # pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    # frameRate = 1000/ds.FrameTime
    nbrOfFrames = ds.NumberOfFrames
    im1 = ds.pixel_array[1]
    if False:
        plt.imshow(im1)
        plt.show()
        wait = input("Press enter to continue")
    
    frame = (patient[3],patient[4])
    for i in range(1,int(nbrOfFrames/2)):
        im = ds.pixel_array[i]
        f_height = np.array(range(frame[0][0], frame[0][1]))
        f_width = np.array(range(frame[1][0],frame[1][1]))
        focused_im = im[:][f_height[:, None],f_width]
        cv2.imwrite(output_dir + patient[2] + str(nbr) + "_extra" + str(i) + ".png",focused_im)

for patient in patients:
    output_dir = "dataset/extra_testdata/" + patient[2] + "/"
    try:
        os.mkdir(output_dir)
    except OSError:
        print("Folder exists")
    for j in range(patient[0],patient[1]):
        generate_img(j,output_dir)


