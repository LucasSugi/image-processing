# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Restauração de Imagens 
'''

import numpy as np
import imageio as img

#Filter with local redction
def adaptiveLocalNoiseReductionFilter(imageNoisy,sizeFilter,varianceNoisy): 
    
    #Size of image noisy
    M,N = imageNoisy.shape
      
    #wrap image
    increase = int(sizeFilter/2)
    wrap = np.pad(imageNoisy,increase,'wrap') 
    
    #Extract all wrap images 
    res = np.zeros([M,N,sizeFilter,sizeFilter])
    for i in range(M):
        for j in range(N):
            res [i][j] = wrap[i:i+sizeFilter,j:j+sizeFilter]

    #Vector with mean and variance
    variance = np.var(res,axis=(2,3))
    mean = np.mean(res,axis=(2,3))
    
    #aply filter
    imageNoisy = imageNoisy - (varianceNoisy/variance)*(imageNoisy-mean)
    
    return imageNoisy

#Step B in filter with medium
def stepB(Iout,imageNoisy,zmed,zmax,zmin,i,j):
    
    B1 = imageNoisy[i][j]-zmin
    B2 = zmed-zmax
    if(B1>0 and B2 <0):
        Iout[i][j] = imageNoisy[i][j]
    else:
        Iout[i][j] = zmed

#Step A in filter with medium
def stepA(Iout,imageNoisy,sizeFilter,paramFilter,i,j):
    
    #symImage image
    increase = int(sizeFilter/2)
    symImage = np.pad(imageNoisy,increase,'symmetric')
    
    zmed = np.median(symImage[i:i+sizeFilter,j:j+sizeFilter])
    zmax = np.max(symImage[i:i+sizeFilter,j:j+sizeFilter])
    zmin = np.min(symImage[i:i+sizeFilter,j:j+sizeFilter])
    A1 = zmed-zmin
    A2 = zmed-zmax
    
    if(A1>0 and A2<0):
        stepB(Iout,imageNoisy,zmed,zmax,zmin,i,j)
    else:
        if(sizeFilter+1 <= paramFilter):
            stepA(Iout,imageNoisy,sizeFilter+1,paramFilter,i,j)
        else:
            Iout[i][j] = zmed

#Filter with medium
def mediumAdaptiveFilter(imageNoisy,sizeFilter,paramFilter):

    #Size of image noisy
    M,N = imageNoisy.shape
    
    #Creating the out image
    Iout = np.zeros([M,N])

    for i in range(M):
        for j in range(N):
            stepA(Iout,imageNoisy,sizeFilter,paramFilter,i,j)      
               
    return Iout

#Filter using media counter harmonic
def counterHarmonicMeanFilter(imageNoisy,sizeFilter,paramFilter):
    
    #Size of image
    M,N = imageNoisy.shape

    #Creating out image
    Iout = np.zeros([M,N])
    
    #Constant image
    imageNoisy = np.pad(imageNoisy,int(sizeFilter/2),'constant')
    
    #Apply filter
    for i in range(M):
        for j in range(N):
            g = imageNoisy[i:i+sizeFilter,j:j+sizeFilter]
            Iout[i][j] = np.sum(np.power(g[g!=0],paramFilter+1))/np.sum(np.power(g[g!=0],paramFilter))

    return Iout

#Computes the error between the two images
def error(original, genereted):

    #limits image
    M,N = original.shape 
    
    errorSum = ((np.power(original-genereted,2))/(M*N)).sum()
    errorSum = np.sqrt(errorSum)
    print(errorSum)

#Read the data from input
def inputData():
    
    #read the names of original and noisy images
    nameImageOriginal = str(input()).rstrip()
    nameImageNoisy = str(input()).rstrip()

    #filter option
    filterOption = int(input())

    #size of filter
    sizeFilter = int(input())

    #parameter of filter
    paramFilter = float(input())
    
    #read the two images
    imageOriginal = img.imread(nameImageOriginal);
    imageNoisy = img.imread(nameImageNoisy);
    
    #Choice the rigth filter
    if(filterOption == 1):
        imageNoisy = adaptiveLocalNoiseReductionFilter(imageNoisy,sizeFilter,np.power(paramFilter,2))
    elif(filterOption == 2):
        imageNoisy = mediumAdaptiveFilter(imageNoisy,sizeFilter,int(paramFilter))
    else:
        imageNoisy = counterHarmonicMeanFilter(imageNoisy,sizeFilter,paramFilter)
   
    #Computes the error 
    error(imageOriginal,imageNoisy)

#call for read data from input
inputData();
