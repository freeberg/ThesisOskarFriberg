import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import sqrt, pi, cos, sin
# from skimage import feature
from collections import defaultdict


def find_circle_roi(img_path, rmin=35, rmax=38, steps=100, thresh=0.3):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    points = np.zeros((steps * (rmax - rmin + 1),3))
    for t in range(steps):
        for r in range(rmin, rmax+1):
            points[t * (rmax+1 - rmin) + (r-rmin)] = [r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))]

    can_img = cv2.Canny(img, 20, 40)
    plt.imshow(can_img,cmap = 'gray')
    plt.show()
    acc = defaultdict(int)
    x_edges, y_edges = np.where(can_img == 255)
    edges = [(x, y) for (x,y) in zip(x_edges, y_edges)]
    for x, y in edges:
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= thresh and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            # print(v / steps, x, y, r)
            circles.append((x, y, r))
    for x, y, r in circles:
        plt.imshow(can_img,cmap = 'gray')
        seg_circle = plt.Circle((x, y), r, color='r')
        seg_circle.set_fill(False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.add_artist(seg_circle)
        plt.show()
        wait = input("Press enter to continue")
    
    plt.show() 
    print(circles)

find_circle_roi("../dataset/magSRAD/Magnus42he2.png")
    