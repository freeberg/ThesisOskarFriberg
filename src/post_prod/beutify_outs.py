import numpy as np
import cv2
import os

magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (155, 187, "FP2",(100,500), (200, 600))
patients = [magnus, FP1, FP2]

def mix_im_n_mask(impath, maskpath, truepath):
    im = cv2.imread(impath)
    mask = cv2.imread(maskpath)
    true = cv2.imread(truepath)
    h, w, c = im.shape
    mix = np.zeros(im.shape)
    for i in range(h):
        for k in range(w):
            if int(true[i][k][0]) > 0 and int(mask[i][k][0]) > 0:
                mix[i][k] = im[i][k] + [int(true[i][k][0])/2, 0, 0]
            else:
                mix[i][k] = im[i][k] + [0, int(true[i][k][0])/2, int(mask[i][k][0])/2]
    
    return mix

def beutify_outputs(out_dir, imgs_dir, seg_dir, exp_dir):
    for patient in patients:
        print("Beutify " + patient[2])
        imgs_dir_p = imgs_dir + patient[2] + "/"
        exp_dir_p = exp_dir + patient[2] + "/"
        out_dir_p = out_dir + patient[2] + "/"
        if not os.path.exists(out_dir_p):
                os.makedirs(out_dir_p)

        for i in range(patient[1]-patient[0]):
            im_name = patient[2] + str(patient[0] + i) 
            im_path = imgs_dir_p + im_name + ".png"
            exp_path = exp_dir_p + im_name + "_seg.png"
            seg_path = seg_dir + im_name + "_mask.png"
            mix = mix_im_n_mask(im_path, exp_path, seg_path)
            cv2.imwrite(out_dir_p + "/" + patient[2] + str(i + patient[0]) + '_fancy.png', mix)


out_dir = "project_ims/"
imgs_dir = "dataset/c3/"
seg_dir = "dataset/masks/"
exp_dir = "F:/Thesis/best_exp/"
beutify_outputs(out_dir, imgs_dir, seg_dir, exp_dir)
