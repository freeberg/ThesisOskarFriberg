import pydicom as dicom
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import csv

magnus = (42, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 169, "Tobias", (136,488), (186,660))
roger = (223, 284, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (105, 187, "FP2",(100,500), (200, 600))
patient = FP1
seg_dict = []
with open("src/csvfiles/manuellSegResultat"+ patient[2] + ".csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        seg_dict.append((row[0], float(row[1]), float(row[2]), float(row[3])))

manual=np.array(seg_dict) #[["043", 374.5587, 464.2196, 5640], ["044", 374.3383, 462.5943, 5640], ["045", 376.7325, 463.2172, 5660], ["046", 374.2353, 464.6563, 5650], ["047", 371.7647, 464.0414, 5640], ["048", 376, 467.2214, 5640], ["049", 371.8677, 462.7804, 5650], ["050", 374.2675, 459.8425, 5650], ["051", 374.5, 459.6563, 5640], ["052", 376.5, 459.0286, 5660], ["053", 377.7647, 457.2816, 5650], ["054", 379.397, 454.031, 5640], ["055", 380.6338, 453.031, 5630], ["056", 382.8727, 454.5918, 5630], ["057", 380.706, 447.9069 ,5640], ["058", 379.8677, 445.2196, 5620], ["059", 376.0737, 442.4678, 5620], ["060", 377.7039, 438.5298, 5630], ["061", 378.294, 438.3358 ,5640], ["062", 380.294, 437.0243 ,5640], ["063", 385.7203, 434.1557, 5640], ["064", 384.4557, 426.4702, 5640], ["065", 390.9707, 430.3461, 5620], ["066", 390.0701, 432.7232, 5620], ["067", 388.8312, 433.4672, 5640], ["068", 391.8233, 430.9069, 5620], ["069", 389.9263, 430.2196, 5630]])
imNrs = manual[:,0]
yCoor = [float(i) for i in manual[:,1]]
xCoor = [float(i) for i in manual[:,2]]
diams = [float(i) for i in manual[:,3]]

seg_traing_set = {}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

savOrShow = "sa"

for i in range(int(imNrs[-1])-int(imNrs[0])+ 1):
    if patient == FP2 and "154" in imNrs[i]:
        continue
    
    ds = dicom.dcmread('/home/friberg/Programming/ThesisOskarFriberg/dataset/' + patient[2] + '/IM_0' + imNrs[i])

    rows = ds.Rows
    cols = ds.Columns
    pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    image = rgb2gray(ds.pixel_array[0][:][:])
    dia_pix_w = pixelmmWidth * 1/2000.*diams[i]
    dia_pix_h = pixelmmHeight * 1/2000.*diams[i]
    crop_x, crop_y = (yCoor[i] - patient[3][0], xCoor[i] - patient[4][0])
    seg_circle = plt.Circle((crop_y, crop_x), dia_pix_h, color='w')
    seg_circle.set_fill(True)
    f_height = np.array(range(patient[4][0], patient[4][1]))
    f_width = np.array(range(patient[3][0], patient[3][1]))
    if('h' in savOrShow):
        corp_img = np.array(image[f_width[:, None],f_height], dtype=np.uint8)
        
        ax = plt.gca()

        ax.add_artist(seg_circle)

        plt.show()
        wait = input("Press enter to continue")
    if('c' in savOrShow):

        background = np.zeros((len(f_width), len(f_height)))
        cv2.circle(background, (int(crop_y), int(crop_x)), int(dia_pix_h), [255,0,0], -1)
        cv2.imwrite(patient[2] + imNrs[i] + "_mask.png", background)
        # wait = input("Press enter to continue")
    else:
        seg_traing_set[imNrs[i]] = (crop_x, crop_y, dia_pix_w)


if('a' in savOrShow):
    str_add = "csvfiles/seg_training_data_"
    w = csv.writer(open(str_add+patient[2]+".csv", "w", newline=''), delimiter=';', quoting=csv.QUOTE_MINIMAL)
    for key, val in seg_traing_set.items():
        w.writerow([key, val])   
