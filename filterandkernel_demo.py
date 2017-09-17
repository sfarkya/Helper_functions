# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:15:40 2017

@author: saurabh
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt
#from scipy.signal import convolve2d as conv2d
from customHistogram import histogram
from mulKernel import mulkernel

## Reading an image. 
img = cv2.imread('reflection_test.jpg');
# IN opencv, img is read as numpy array in reverse order. 
# So, to correct it we use this operation. 
# So, BGR to RGB conversion takes place. 

# Plot image 
plt.subplot(441)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
plt.title('RGB image')

# Histogram 
plt.subplot(442)
histogram(img)
plt.title('Histogram')

# Grayscale image 
plt.subplot(443)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray,cmap='gray')
plt.title('Grayscale image')

# X-derivative 
plt.subplot(444)
x_kernel = (float(1)/float(3))*np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
mulkernel(img,x_kernel,'X_derivative')


# Y-derivative
plt.subplot(445)
y_kernel = (float(1)/float(3))*np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
mulkernel(img,x_kernel,'Y_derivative')

# Gaussian Smoothing 
plt.subplot(446)
g_kernel = np.array([[.01,.08,.01],[.08,.64,.08],[.01,.08,.01]])
mulkernel(img,g_kernel,'Gaussian')

# X_derivative_Gaussian
plt.subplot(447)
dxg_kernel = np.array([[.05,0,-.05],[.34,0,-.34],[.05,0,-.05]])
mulkernel(img,dxg_kernel,'Xd_Gaussian')

# Y_derivative_Gaussian
plt.subplot(448)
dyg_kernel = np.array([[.05,.34,.05],[0,0,0],[-.05,-.34,-.05]])
mulkernel(img,dyg_kernel,'Yd_Gaussian')


# Laplacian 
plt.subplot(449)
lap_kernel = np.array([[.3,.7,.7],[0.7,-4,.7],[.3,.7,.3]])
mulkernel(img,lap_kernel,'Laplacian')

# Vertical sobel filter
plt.subplot(4,4,10)
vsobel_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
mulkernel(img,vsobel_kernel,'Vertical Sobel Filter')

#Horizontal sobel filter 
plt.subplot(4,4,11)
hsobel_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
mulkernel(img,hsobel_kernel,'Horizontal Sobel Filter')

plt.show()


