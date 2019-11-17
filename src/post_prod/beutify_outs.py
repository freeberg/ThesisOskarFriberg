import numpy as np
import cv2
import os

magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (155, 187, "FP2",(100,500), (200, 600))
patients = [roger] #, magnus]

def mix_im_n_mask(impath, maskpath, truepath):
    im = cv2.imread(impath)
    mask = cv2.imread(maskpath)
    true = cv2.imread(truepath)
    h, w, c = im.shape
    mix = np.zeros(im.shape)
    print(mask.shape)
    for i in range(h):
        for k in range(w):
            mix[i][k] = im[i][k] + [int(true[i][k][0]), int(mask[i][k][0])/3, 0]
    
    return mix

def beutify_outputs(out_dir, imgs_dir, seg_dir, exp_dir):
    for patient in patients:
        imgs_dir = imgs_dir + patient[2] + "_c3/"
        seg_dir = seg_dir + patient[2] + "masks/"
        exp_dir = seg_dir + patient[2]
        dir_p = out_dir + patient[2]
        if not os.path.exists(dir_p):
                os.makedirs(dir_p)

        for i in range(patient[1]-patient[0]):
            im_name = patient[2] + str(patient[0] + i) +  ".png"
            im_path = imgs_dir + im_name 
            exp_path = exp_dir + im_name
            seg_path = seg_dir + im_name
            mix = mix_im_n_mask(im_path, exp_path, seg_path)
            cv2.imwrite(dir_p + "/" + str(i + patient[0]) + '.png', mix)


out_dir = "project_ims/"
imgs_dir = "dataset/raw/"
seg_dir = "dataset/raw/"
exp_dir = "best_exp/"
beutify_outputs(out_dir, imgs_dir, seg_dir, exp_dir)
