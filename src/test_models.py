import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from UNet.generate_test_models import gen_test_models
from UNet.predict import mask_to_image, predict_img
from UNet.unet import UNet
from UNet.utils import get_ids, plot_img_and_mask

from evaluate_masks import evaluate_masks


def latest_cp(ckpnts):
    cps = [f for f in os.listdir(ckpnts)]
   # print(ckpnts)
    cps.remove("itr_loss.txt")
    cps.sort()
    return cps[-1]

def fancify_result(result_dir):
    magnus = str(result_dir["Magnus"])
    roger = str(result_dir["Roger"])
    FP1 = str(result_dir["FP1"])
    FP2 = str(result_dir["FP2"])
    total = str(result_dir["Total"])

    return "Total: " + total + "    Magnus: " + magnus + "    Roger: " + roger + "    FP1: " + FP1 + "    FP: " + FP2

gpu = True
if not gpu:
    print("Not using GPU, will take long time!")

print("Generate models !! Will take some time")
gen_test_models(gpu)

viz = False
scale = [0.75, 1]
out_thresh = [0.2, 0.1]
crf = True

model_dir = "checkpoints/"
models = [f + "/" for f in os.listdir(model_dir)]
test_dir = "dataset/test/"

for sc in scale:
    for thr in out_thresh:
        for m in models:
            ids = get_ids(test_dir)
            white_area = 0

            out_dir = "experiments/"
            curr_dir = out_dir + m
            cp = latest_cp(model_dir + m)

            curr_dir = curr_dir + "sc" + str(sc) + "_thresh" + str(thr) + "/"
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)

            net = UNet(n_channels=3, n_classes=1)

            if gpu:
                print("Using CUDA version of the net, prepare your GPU !")
                print(model_dir + m + cp)
                net.cuda()
                net.load_state_dict(torch.load(model_dir + m + cp))
            else:
                net.cpu()
                net.load_state_dict(torch.load(model_dir + m + cp, map_location='cpu'))
                print("Using CPU version of the net, this may be very slow")

            print(m + " loaded !")


            for i in ids:
                full_img = Image.open(test_dir + i + ".png")
                mask = predict_img(net, full_img, sc, thr, crf, gpu)
                if viz:

                    fig = plt.figure()
                    a = fig.add_subplot(1, 2, 1)
                    a.set_title('Input image')
                    plt.imshow(full_img)
                    b = fig.add_subplot(1, 2, 2)
                    b.set_title('Output mask')
                    plt.imshow(mask)
                    plt.savefig(curr_dir + i + '_diff.png')
                    plt.close(fig) 

                mask_img = np.array([[255 * int(j) for j in i] for i in mask])

                cv2.imwrite(curr_dir + i + '_seg.png', mask_img)
            result_dir = evaluate_masks(curr_dir)
            
            fd = open(out_dir + "dice_values.txt", "a+")
            s = ""
            for key,val in result_dir.items():
                s = s + key + ": " + str(val) + " | "
            fd.write(m + " Dice Values: " + s + "\n")
            fd.close()

            print("Experiment with model " + m + "sc" + str(sc) + "_thresh" + str(thr) + " gave dice " + str(result_dir["Total"]))
