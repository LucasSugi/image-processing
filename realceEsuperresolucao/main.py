# -*- coding: utf-8 -*-

import numpy as np
import imageio as img
import math

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Realce e Superresolução 
'''

#Calculates the histogram of each image
def individualTransfer(im1,im2,im3,im4):
   
    #max pixel in an image
    maxPixel = 256 

    #limits image
    M = im1.shape[0]
    N = im1.shape[1]

    #vector that has a histogram from image
    hist1 = np.zeros(maxPixel,int)
    hist2 = np.zeros(maxPixel,int)
    hist3 = np.zeros(maxPixel,int)
    hist4 = np.zeros(maxPixel,int)
    
    #populating the histogram vector
    for i in range(maxPixel):
            hist1[i] = (im1==i).sum()
            hist2[i] = (im2==i).sum()
            hist3[i] = (im3==i).sum()
            hist4[i] = (im4==i).sum()
    
    #calculating the cumulated histogram
    for i in range(1,maxPixel):
        hist1[i] = hist1[i] + hist1[i-1]
        hist2[i] = hist2[i] + hist2[i-1]
        hist3[i] = hist3[i] + hist3[i-1]
        hist4[i] = hist4[i] + hist4[i-1]
        
    #Multiplicative factor
    multiplicativeFactor = (maxPixel-1)/(M*N)
    
    #new matrix of new image
    newIm1 = np.zeros((M,N),int)
    newIm2 = np.zeros((M,N),int)
    newIm3 = np.zeros((M,N),int)
    newIm4 = np.zeros((M,N),int)

    #equalizing
    for i in range(maxPixel):
        s1 = hist1[i] * multiplicativeFactor
        s2 = hist2[i] * multiplicativeFactor
        s3 = hist3[i] * multiplicativeFactor
        s4 = hist4[i] * multiplicativeFactor

        newIm1[np.where(im1 == i)] = s1
        newIm2[np.where(im2 == i)] = s2
        newIm3[np.where(im3 == i)] = s3
        newIm4[np.where(im4 == i)] = s4
    
    return newIm1,newIm2,newIm3,newIm4

#Calculates the histogram of all image together
def jointTransfer(im1,im2,im3,im4):
    
    #max pixel in an image
    maxPixel = 256 

    #limits image
    M = im1.shape[0]
    N = im1.shape[1]

    #vector that has a histogram from image
    hist = np.zeros(maxPixel,int)
    
    #populating the histogram vector
    for i in range(maxPixel):
            hist[i] = (im1==i).sum()
            hist[i] = (im2==i).sum()
            hist[i] = (im3==i).sum()
            hist[i] = (im4==i).sum()
    
    #calculating the cumulated histogram
    for i in range(1,maxPixel):
        hist[i] = hist[i] + hist[i-1]

    #Multiplicative factor
    multiplicativeFactor1 = im1.max()/(M*N)
    multiplicativeFactor2 = im2.max()/(M*N)
    multiplicativeFactor3 = im3.max()/(M*N)
    multiplicativeFactor4 = im4.max()/(M*N)
    
    #calculating the mean
    hist = hist/4
    
    #new matrix of new image
    newIm1 = np.zeros((M,N),int)
    newIm2 = np.zeros((M,N),int)
    newIm3 = np.zeros((M,N),int)
    newIm4 = np.zeros((M,N),int)
    
    #equalizing
    for i in range(maxPixel):
        s1 = hist[i] * multiplicativeFactor1
        s2 = hist[i] * multiplicativeFactor2
        s3 = hist[i] * multiplicativeFactor3
        s4 = hist[i] * multiplicativeFactor4

        newIm1[np.where(im1 == i)] = s1
        newIm2[np.where(im2 == i)] = s2
        newIm3[np.where(im3 == i)] = s3
        newIm4[np.where(im4 == i)] = s4

    return im1,im2,im3,im4

#applies the gamma function in image
def gammaSetting(im,parameterE):
    
    M = im.shape[0]
    N = im.shape[1]
    parameterE = 1/parameterE

    #applying the gamma function
    im = np.floor(np.power(im/255.0,parameterE)*255)

    return im

#calculates the super resolution of a image
def superresolution(im1,im2,im3,im4):
    
    #limits image
    M = im1.shape[0]
    N = im1.shape[1]
    
    #matrix of genereted image
    genereted = np.zeros((2*M,2*N))
    
    #creating the genereted image
    for (i,j) in np.ndindex(M,N):
            genereted[i*2][j*2] = im1[i][j]
            genereted[(i*2)+1][j*2] = im2[i][j]
            genereted[i*2][(j*2)+1] = im3[i][j]
            genereted[(i*2)+1][(j*2)+1] = im4[i][j]
    
    return genereted

#calculates the error between the 2 images (original and genereted)
def error(original,genereted):
   
    #limits image
    M = original.shape[0]
    N = original.shape[1]
    
    errorSum = ((np.power(original-genereted,2))/(M*N)).sum()
    errorSum = math.sqrt(errorSum)
    
    print("{0:.4f}" .format(errorSum))

#Reading data
def inputData():
    
    #reading the base name of images
    baseImage1 = str(input())
    baseImage2 = baseImage1
    baseImage3 = baseImage1
    baseImage4 = baseImage1
    
    #reading the name of high resolution image
    highImage = str(input())

    #enhancement method
    typeMethod = int(input())

    #parameter of enhancement 3
    parameterE = float(input())
    
    #treatment of strings
    baseImage1 = baseImage1[0:len(baseImage1)-1]
    baseImage2 = baseImage2[0:len(baseImage2)-1]
    baseImage3 = baseImage3[0:len(baseImage3)-1]
    baseImage4 = baseImage4[0:len(baseImage4)-1]
    highImage = highImage[0:len(highImage)-1]

    #read images that will be used for histogram equalization
    baseImage1 +="1.png"
    baseImage2 +="2.png"
    baseImage3 +="3.png"
    baseImage4 +="4.png"
    im1 = img.imread(baseImage1)
    im2 = img.imread(baseImage2)
    im3 = img.imread(baseImage3)
    im4 = img.imread(baseImage4)
    
    #read the high image
    highImage += ".png"
    im5 = img.imread(highImage)
    
    #verify the type of enhancement
    if typeMethod == 1:
        im1,im2,im3,im4 = individualTransfer(im1,im2,im3,im4) 
    elif typeMethod == 2:
        im1,im2,im3,im4 = jointTransfer(im1,im2,im3,im4)
    elif typeMethod == 3:
        im1 = gammaSetting(im1,parameterE)
        im2 = gammaSetting(im2,parameterE)
        im3 = gammaSetting(im3,parameterE)
        im4 = gammaSetting(im4,parameterE)
    
    #applies the superresolution
    genereted = superresolution(im1,im2,im3,im4)

    #calculates the error
    error(im5,genereted)

#called for inputData
inputData()
