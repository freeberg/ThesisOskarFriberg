import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2



ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset/DICOM 3D/Magnus/IM_0002')

# print(ds.pixel_array)
print(ds.pixel_array.shape)
im1 = ds.pixel_array[1]
print(im1.shape)
plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
plt.colorbar()
plt.show()