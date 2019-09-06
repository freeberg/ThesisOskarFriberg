import os, sys
from .unet import UNet
from .train import train_net


def gen_test_models():
    lr_opts = [0.05, 0.1, 0.2]
    batch_opts = [1, 2]#, 4]
    scale_opts = [0.75, 1]
    optimzer_opts = ["ADAM", "ADAGRAD", "SGD"]
    gpu = True
    
    for lr in lr_opts:
        for batch in batch_opts:
            for scale in scale_opts:
                for optimzer in optimzer_opts:
                    net = UNet(n_channels=3, n_classes=1)
                    if gpu:
                        net.cuda()
                    train_net(net=net,
                      epochs=5,
                      batch_size=batch,
                      lr=lr,
                      gpu=gpu,
                      img_scale=scale,
                      optimzer_opt=optimzer)

