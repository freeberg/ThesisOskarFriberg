import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def srad(img, ):
    I = np.sum(np.sum(image))
    I2 = np.sum(np.sum(image * image))









# function [image] = srad_core(image,NeROI,Nc,Nr,iN,iS,jW,jE,dN,dS,dW,dE,c,lambda,niter)
# % Core computation

# for iter = 1:niter
#     tot = sum(sum(image));
#     tot2= sum(sum(image.*image));
#     meanROI = tot / NeROI;
#     varROI  = (tot2 / NeROI) - meanROI * meanROI;
#     q0sqr   = varROI / (meanROI * meanROI);
#     for j = 1:Nc
#         for i = 1:Nr
#             k = i + Nr * (j - 1);
#             Jc = image(k);
#             % directional derivates (every element of IMAGE)
#             dN(k) = image(iN(i) + Nr * (j-1)) - Jc;
#             dS(k) = image(iS(i) + Nr * (j-1)) - Jc;
#             dW(k) = image(i + Nr * (jW(j)-1)) - Jc;
#             dE(k) = image(i + Nr * (jE(j)-1)) - Jc;
            
#             G2 = (dN(k) * dN(k) + dS(k) * dS(k) ...
#                 + dW(k) * dW(k) + dE(k) * dE(k)) / (Jc * Jc);
#             L  = (dN(k) + dS(k) + dW(k) + dE(k)) / Jc;
#             num  = (0.5 * G2) - ((1.0 / 16.0) * (L * L)) ;
#             den  = 1 + (.25*L);
#             qsqr = num / (den * den);
#             den  = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr)) ;
#             c(k) = 1.0 / (1.0 + den);
#             if c(k) < 0
#                 c(k) = 0;
#             elseif c(k) > 1
#                 c(k) = 1;
#             end
#         end
#     end
#     for j = 1:Nc
#         for i = 1:Nr
#             k = i + Nr * (j - 1);
#             cN = c(k);
#             cS = c(iS(i) + Nr *(j-1));
#             cW = c(k);
#             cE = c(i + Nr * (jE(j)-1));
#             D = cN * dN(k) + cS * dS(k) + cW * dW(k) + cE * dE(k);
#             image(k) = image(k) + 0.25 * lambda * D;
#         end
#     end
# end

# end