
from train import train_net
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from skimage import data, exposure, io

# path = "data/train_masks/0cdf5b5d0ce1_02_mask.gif"
# # im = cv2.imread(path)
# im = io.imread(path)

# print(min(im[:][:][:]))
# print(max(im))







net = UNet(n_channels=3, n_classes=1)

# net.load_state_dict(torch.load("MODEL.pth"))

# net.cuda()


train_net(net,1,1,0.1,0.05,False,False,0.1)
