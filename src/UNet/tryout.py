"FIXA PATH, OM DU VILL KÖRA!"
from UNet.train import train_net
from UNet.unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import cv2

net = UNet(n_channels=3, n_classes=1)

# net.load_state_dict(torch.load("MODEL.pth"))

# net.cuda()


train_net(net,1,1,0.1,0.05,False,False,0.1)
