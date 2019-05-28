import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def srad(image, rows, cols, it, lamb):

    i_north = np.array([i-1 for i in range(rows+1)])
    i_south = np.array([i+1 for i in range(rows)])
    j_west = np.array([i-1 for i in range(cols+1)])
    j_east = np.array([i+1 for i in range(cols)])

    d_north = np.zeros((rows, cols))
    d_south = np.zeros((rows, cols))
    d_west = np.zeros((rows, cols))
    d_east = np.zeros((rows, cols))

    c  = np.zeros((rows, cols))

    # set boundary value
    i_north[0] = 0
    i_south[-1] = rows - 1
    j_west[0] = 0
    j_east[-1] = cols - 1

    imsize = cols * rows

    for t in range(it):
        I = np.sum(np.sum(image))
        I2 = np.sum(np.sum(image * image))
        mean = I / imsize
        var = (I2 / imsize) - mean**2
        sqr_q0 = var / (mean**2)
        for j in range(cols):
            for i in range(rows):
                #i [j]= i + rows * j
                Jc = image[i][j] # Energy
                # directional derivates (every element of IMAGE)
                d_north[i][j] = image[i_north[i]][j] - Jc
                d_south[i][j] = image[i_south[i]][j] - Jc
                d_west[i][j] = image[i][j_west[j]] - Jc
                d_east[i][j] = image[i][j_east[j]] - Jc

                GraRI_magn_2 = (d_north[i][j] * d_north[i][j] + d_south[i][j] * d_south[i][j] + d_west[i][j] * d_west[i][j] + d_east[i][j] * d_east[i][j]) / (Jc * Jc)
                GraLI_2  = (d_north[i][j] + d_south[i][j] + d_west[i][j] + d_east[i][j]) / Jc
                num  = (0.5 * GraRI_magn_2) - ((1.0 / 16.0) * (GraLI_2 * GraLI_2)) 
                den  = 1 + (.25*GraLI_2)
                sqr_q = num / (den * den)
                den  = (sqr_q - sqr_q0) / (sqr_q0 * (1 + sqr_q0)) 
                c[i][j] = 1.0 / (1.0 + den)
                if c[i][j] < 0:
                    c[i][j] = 0
                elif c[i][j] > 1:
                    c[i][j] = 1
        for j in range(cols):
            for i in range(rows):
                #i [j]= i + rows * j
                c_north = c[i][j]
                c_south = c[i_south[i]][j]
                c_west = c[i][j]
                c_east = c[i][j_east[j]]
                D = c_north * d_north[i][j] + c_south * d_south[i][j] + c_west * d_west[i][j] + c_east * d_east[i][j]
                
                image[i][j] = image[i][j] + 0.6 * lamb * D
    
    return image
