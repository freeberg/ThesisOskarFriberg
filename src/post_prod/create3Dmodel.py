import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2

distance = 0.3 # Distance between each seg is 300µm
magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (155, 187, "FP2",(100,500), (200, 600))
patients = [roger] #, magnus]

"""
1. Get info of US data - distances from each seg, (Dicom-pixelMM) 
3. Create a room matrix, calculate the length of artery and thickness
2. ?? Normalize the segments with the fist one?? 
4. Place the segments in the room
5. Connect the segments, either by all points or edges (Think all is easier)
6. Create the 3D plot
7. Profit
"""






def get_dicom_info(patient, imNbr):
    '''in:patient triple, in:imNbr int'''
    if patient == magnus or patient == FP1:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_00'+str(imNbr)
    else:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_0'+str(imNbr)
    ds = dicom.dcmread(im_path)
    pixelmm = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    im1 = np.zeros((rows, cols), np.float32)
    im1[:][:] = rgb2gray(ds.pixel_array[0][:][:])
    f_height = np.array(range(patient[3][0], patient[3][1]))
    f_width = np.array(range(patient[4][0], patient[4][1]))
    focused_img = np.array(im1[:][f_height[:, None],f_width], dtype=np.uint8)
    return focused_img, len(f_height), len(f_width)

