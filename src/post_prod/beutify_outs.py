import numpy as np
import cv2

magnus = (43, 70, "Magnus", (100,500), (200, 600))
tobias = (109, 168, "Tobias", (136,488), (186,660))
roger = (223, 283, "Roger", (100,500), (200, 600))
FP1 = (18, 99, "FP1",(100,500), (200, 600))
FP2 = (155, 187, "FP2",(100,500), (200, 600))
patients = [roger] #, magnus]

def mix_im_n_mask(impath, maskpath):
    im = cv2.imread(impath)
    mask = cv2.imread(maskpath)
    h, w, c = im.shape
    mix = np.zeros(im.shape)
    print(mask.shape)
    for i in range(h):
        for k in range(w):
            mix[i][k] = im[i][k] + [0,0,int(mask[i][k][0])/3]
    
    return mix

def beutify_outputs(out_dir, imgs_dir, seg_dir):
    for patient in patients:
        imgs_dir = imgs_dir + patient[2] + "_c3/"
        seg_dir = seg_dir + patient[2] + "masks/"
        dir_p = out_dir + patient[2]
        if not os.path.exists(dir_p):
                os.makedirs(dir_p)

        for i in range(patient[1]-patient[0]):

            mask_path = "dataset/train_masks/Magnus043_mask.png"
            im_path = "dataset/train/Magnus043.png"
            mix = mix_im_n_mask(im_path, mask_path)
            cv2.imwrite(dir_p + "/" + str(i + patient[0]) + '.png', mix)


out_dir = "project_ims/"
imgs_dir = "dataset/raw/"
seg_dir = "experiments/"
beutify_outputs(out_dir, imgs_dir, seg_dir)
