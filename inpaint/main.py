# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Inpainting 
'''

import numpy as np
import imageio as img

#apply the method of Gerchberg Papoulis to restore the image
def gerchbergPapoulis(imgd,imgm,T):
    
    #apply the fourier series
    M = np.fft.fft2(imgm)
    
    #filter 1
    filter1 = 0.9*M.max()

    #creating a filter of mean that will be used in convolution
    H,W = imgd.shape
    meanFilter = np.zeros([H,W])
    for i in range(7):
        for j in range(7):
            meanFilter[i][j] = 1/49
    meanFilter = np.fft.fft2(meanFilter)
    
    #iterations to get a better image
    G = imgd
    for i in range(T):
        G = np.fft.fft2(G)
        
        #apply filter 
        filter2 = 0.01*G.max()
        for j in range(H):
            for k in range(W):
                if( G[j][k] >= filter1 and G[j][k] <= filter2):
                    G[j][k] = 0

        #convolution
        G = np.multiply(G,meanFilter)

        #get the inverse transform
        G = np.real(np.fft.ifft2(G))

        #normalizing
        G = np.uint8(((G-G.min())/(G.max()-G.min()))*255)
        
        #insert pixels
        G = np.multiply((1-(imgm/255)),imgd) + np.multiply((imgm/255),G)
    
    return G

#computes the error between the two images
def error(original, genereted):

    #limits image
    M,N = original.shape 
    original = np.uint8(original)
    genereted = np.uint8(genereted)
    
    errorSum = ((np.power(original-genereted,2))/(M*N)).sum()
    errorSum = np.sqrt(errorSum)
    print("{0:.5f}" .format(errorSum))

#read the data
def inputData():

    #name of original image
    nameOriginalImage = str(input()).rstrip()
    
    #name of deteriorated image
    nameDeterioratedImage = str(input()).rstrip()
    
    #name of mask image
    nameMaskImage = str(input()).rstrip()
    
    #number of iterations
    T = int(input())

    #read all images
    imgo = img.imread(nameOriginalImage)
    imgd = img.imread(nameDeterioratedImage)
    imgm = img.imread(nameMaskImage)
    
    #call for Gerchberg Papoulis method
    G = gerchbergPapoulis(imgd,imgm,T)
    
    #calculates the error
    error(imgo,G)

#call for read of data
inputData();
