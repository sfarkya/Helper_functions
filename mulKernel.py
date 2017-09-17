# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 00:38:30 2017

@author: saurabh
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def mulkernel(img,k,title): 
    temp_img = np.zeros(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(1,img.shape[0]-2):
        for j in range(1,img.shape[1]-2): 
            temp_img[i,j] = np.mean(img[i-1:i+2,j-1:j+2]*k)
      
    #print img.shape, temp_img.shape         
    #plt.subplot(121)
   # plt.imshow(img,cmap='gray') 
    #plt.subplot(122)
    plt.imshow(temp_img,cmap='gray')
    plt.title(title)
    #plt.show()