import pywt
import numpy as np
from scipy.signal import convolve2d
import math
import cv2

def downscale(image, scale_row, scale_col):
    if scale_row != 1:
        return image[1::2]
    elif scale_col != 1:
        image = np.transpose(image)
        image = image[1::2]
        return np.transpose(image)

def row_wise_convolve(image, filterr):
    convolved_img = []
    for i in range(len(image)):
        convolved_img.append(list(np.convolve(image[i], filterr)))
    return convolved_img

def col_wise_convolve(image, filterr):
    image = np.transpose(image)
    image = row_wise_convolve(image,filterr)
    return np.transpose(image)

def dwt(image, lpf, hpf):
    #row wise convolution
    lowPassConvolved = row_wise_convolve(image,lpf)
    highPassConvolved = row_wise_convolve(image,hpf)

    print(lowPassConvolved[0])
    #downscaling
    l = downscale(lowPassConvolved, 1, 2)
    h = downscale(highPassConvolved, 1, 2)
    
    #column wise convolution
    l_lowPassConvolved = col_wise_convolve(l,lpf)
    l_highPassConcolved = col_wise_convolve(l,hpf)
    h_lowPassConvolved = col_wise_convolve(h,lpf)
    h_highPassConcolved = col_wise_convolve(h,hpf)

    #down scaling
    ll = downscale(l_lowPassConvolved, 2, 1)
    lh = downscale(l_highPassConcolved, 2, 1)
    hl = downscale(h_lowPassConvolved, 2, 1)
    hh = downscale(h_highPassConcolved, 2, 1)

    print(len(hh))
    print(len(hh[0]))
    cv2.imwrite('output/LL.jpg', ll)
    cv2.imwrite('output/LH.jpg', lh)
    cv2.imwrite('output/HL.jpg', hl)
    cv2.imwrite('output/HH.jpg', hh)

img=cv2.imread("input2.jpg",0)
img=cv2.resize(img,(512,512))

lpf = np.array([-0.12940952255092145, 0.22414386804185735,   0.836516303737469, 0.48296291314469025])
hpf = np.array([-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145])
dwt(img,lpf,hpf)