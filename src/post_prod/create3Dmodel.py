import numpy as np
import pydicom as dicom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

distance = 0.3 # Distance between each seg is 300µm
magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (105, 187, "FP2",(100,500), (200, 600))
patients = [roger] #, magnus]

"""
1. Get info of US data - distances from each seg, (Dicom-pixelMM)  - check
3. Create a room matrix, calculate the length of artery and thickness
2. ?? Normalize the segments with the fist one?? 
4. Place the segments in the room
5. Connect the segments, either by all points or edges (Think all is easier)
6. Create the 3D plot
7. Profit
"""

def get_pixelmm(patient, imNbr):
    '''in:patient triple, in:imNbr int'''
    if patient == magnus or patient == FP1:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_00'+str(imNbr)
    else:
        im_path = 'dataset/DICOM/'+patient[2]+'/IM_0'+str(imNbr)
    ds = dicom.dcmread(im_path)
    pixelmm = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    return pixelmm

def get_3D_points(im_mat, nbr, pixelmm):
    z_pos = nbr * (distance * pixelmm)

def plot_3D_artery(patient, seg_dir):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    k = 0
    for i in np.arange(patient[0], patient[1], 1):
        pixelmm = get_pixelmm(patient, i)
        im_mat = cv2.imread(seg_dir + patient[2] + str(i) + "_seg.png",0)
        h, w = im_mat.shape
        X, Y = np.where(im_mat > 0)
        surf = ax.scatter(pixelmm * k, X, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        k = k + 1

# Fixa så att det är test imnbr inte alla!!!
    plt.show()

get_3D_points(magnus, ) 