# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:40:37 2017

@author: saurabh
"""

def histogram(img): # image is of the form BGR. 
    from matplotlib import pyplot as plt 
    
    # Read image 
    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "b",histtype = 'step')
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "g",histtype = 'step')
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "r",histtype = 'step')
    plt.hist(img.ravel(),256,[0,256],color = "k",histtype = 'step')
    plt.show()