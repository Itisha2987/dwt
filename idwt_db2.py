import pywt
import numpy as np
from scipy.signal import convolve2d
import math
import cv2

def upscale(image, scale_row, scale_col):
    upscaled_img = []
    if scale_row != 1:
        for i in range(len(image)):
            upscaled_img.append(image[i])
            upscaled_img.append(np.zeros(len(image[0])))
        return upscaled_img
    elif scale_col != 1:
        image = np.transpose(image)
        for i in range(len(image)):
            upscaled_img.append(image[i])
            upscaled_img.append(np.zeros(len(image[0])))
        np.transpose(upscaled_img)
        return upscaled_img


def row_wise_convolve(image, filterr):
    convolved_img = []
    for i in range(len(image)):
        convolved_img.append(list(np.convolve(image[i], filterr)))
    return convolved_img

def col_wise_convolve(image, filterr):
    image = np.transpose(image)
    image = row_wise_convolve(image,filterr)
    return np.transpose(image)

def idwt(ll, lh, hl, hh, lpf, hpf):

    ll = upscale(ll,2,1)
    lh = upscale(lh,2,1)
    hl = upscale(hl,2,1)
    hh = upscale(hh,2,1)

    ll = col_wise_convolve(ll,lpf)
    lh = col_wise_convolve(lh,hpf)
    l = upscale(ll+lh, 1, 2)
    
    hl = col_wise_convolve(hl,lpf)
    hh = col_wise_convolve(hh,hpf)
    h = upscale(hl+hh, 1, 2)

    l = row_wise_convolve(l,lpf)
    h = row_wise_convolve(h,hpf)
    
    inverse_img = np.array(l+h)
    
    cv2.imwrite('output/inv.jpg', inverse_img)

ll = cv2.imread('output/LL.jpg', 0)
lh = cv2.imread('output/LH.jpg', 0)
hl = cv2.imread('output/HL.jpg', 0)
hh = cv2.imread('output/HH.jpg', 0)

lpf = np.array([0.48296291314469025, 0.836516303737469, 0.22414386804185735, -0.12940952255092145])
hpf = np.array([-0.12940952255092145, -0.22414386804185735, 0.836516303737469, -0.48296291314469025])

idwt(ll,lh,hl,hh,lpf,hpf)