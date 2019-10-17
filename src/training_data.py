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
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (105, 187, "FP2",(100,500), (200, 600))
patients = [FP2]
seg_dict = {}

for patient in patients:
    with open("src/csvfiles/manuellSegResultat"+ patient[2] + ".csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        seg_dict[patient[2]] = []
        for row in reader:
            seg_dict[patient[2]].append([row[0], float(row[1]), float(row[2]), float(row[3])])

seg_traing_set = {}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

savOrShow = "masks"
for patient in patients:
    for i in range(patient[1] - patient[0]):
        print(i)
        im_data = seg_dict[patient[2]][i] # 0:imNbr, 1:yCoor, 2:xCoor, 3:diams
        if patient == FP2 and i == 154:
            continue
        
        ds = dicom.dcmread('dataset/DICOM/' + patient[2] + '/IM_0' + im_data[0])

        rows = ds.Rows
        cols = ds.Columns
        pixelmmWidth = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
        pixelmmHeight = 0.1 / ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
        image = rgb2gray(ds.pixel_array[0][:][:])
        dia_pix_w = pixelmmWidth * 1/2000. * im_data[3]
        dia_pix_h = pixelmmHeight * 1/2000. * im_data[3]
        crop_x, crop_y = (im_data[1] - patient[3][0], im_data[2] - patient[4][0])
        seg_circle = plt.Circle((crop_y, crop_x), dia_pix_h, color='w')
        seg_circle.set_fill(True)
        f_height = np.array(range(patient[4][0], patient[4][1]))
        f_width = np.array(range(patient[3][0], patient[3][1]))
        if('show' in savOrShow):
            corp_img = np.array(image[f_width[:, None],f_height], dtype=np.uint8)
            
            ax = plt.gca()

            ax.add_artist(seg_circle)

            plt.show()
            wait = input("Press enter to continue")
        if('masks' in savOrShow):

            background = np.zeros((len(f_width), len(f_height)))
            cv2.circle(background, (int(crop_y), int(crop_x)), int(dia_pix_h), [255,0,0], -1)
            cv2.imwrite(patient[2] + im_data[0] + "_mask.png", background)
            # wait = input("Press enter to continue")
        else:
            seg_traing_set[im_data[0]] = (crop_x, crop_y, dia_pix_w)


    # if('a' in savOrShow):
    #     str_add = "csvfiles/seg_training_data_"
    #     w = csv.writer(open(str_add+patient[2]+".csv", "w", newline=''), delimiter=';', quoting=csv.QUOTE_MINIMAL)
    #     for key, val in seg_traing_set.items():
    #         w.writerow([key, val])   
