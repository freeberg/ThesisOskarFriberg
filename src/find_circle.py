import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import sqrt, pi, cos, sin
# from skimage import feature
import pydicom as dicom
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import ntpath

def get_roi(img, roe):
    poi = np.where(img != 0)
    if poi[0].size == 0:
        return ((0,0),(0,0))
    min_x = poi[0][0]
    max_x = poi[0][-1]
    min_y = min(poi[1])
    max_y = max(poi[1])
    diff = (max_x - min_x, max_y - min_y)
    roi_x = (min_x - int(roe*diff[0]),max_x + int(roe*diff[0]))
    roi_y = (min_y - int(roe*diff[1]),max_y + int(roe*diff[1]))
    return (roi_x, roi_y)

def get_patient(img_path):
    pat_list = ["Magnus", "Roger", "FP1", "FP2"]
    patient = [x for x in pat_list if x in img_path][0]
    return patient

def guess_radius(img_path):
    fn = ntpath.basename(img_path)
    patient = get_patient(fn)

    if "Magnus" in patient or "FP1" in patient:
        dicom_path = "dataset/" + patient + "/IM_00" + fn[len(patient):(len(patient)+2)]
        ds = dicom.dcmread(dicom_path)
    else:
        dicom_path = "dataset/" + patient + "/IM_0" + fn[len(patient):(len(patient)+3)]
        ds = dicom.dcmread(dicom_path)
    
    pixelmm = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    # Average Male CCA dia is 6.5
    min_r = round(3 * pixelmm)
    max_r = round(3.5 * pixelmm)
    return min_r, max_r


    # https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
def find_circle(img_path, viz=False):
    print(img_path)
    # Load picture and detect edges
    img = cv2.imread(img_path,0)
    roi_x, roi_y = get_roi(img, 0.2)
    if (roi_x[0] == 0) and (roi_x[1] == 0):
        return 0, 0, 0
    image = img_as_ubyte(img[roi_x[0]:roi_x[1], roi_y[0]:roi_y[1]])
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect two radii
    rmin, rmax = guess_radius(img_path)
    hough_radii = np.arange(rmin, rmax, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    if len(cx) == 0:
        return 0, 0, 0
    # Draw them
    if viz:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        img = color.gray2rgb(img)
        co = 1
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(roi_x[0] + center_y, roi_y[0] + center_x, radius,
                                            shape=img.shape)
            img[circy, circx] = (220, 20 * co, 20)
            co = co + 5

        ax.imshow(img, cmap=plt.cm.gray)
        plt.show()

    y_pos = [i + roi_y[0] for i in cx]
    x_pos = [i + roi_x[0] for i in cy]
    return x_pos[0], y_pos[0], radii[0]

find_circle("experiments_SRAD_Cir/model_sc1_lr0.05_batch2_ADAM/sc0.75_thresh0.2/FP195_seg.png", True)
    