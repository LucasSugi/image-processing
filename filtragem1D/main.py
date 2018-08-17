# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Filtragem 1D
'''

import numpy as np
import imageio as img

#calculates the arbitrary filter with spatial domain
def arbitraryFilterSpatialDomain(vectorImg,weights):
    
    #size of vector
    m = vectorImg.shape[0]
    
    #this is used to perform the convolution
    a = int((weights.shape[0]-1)/2)
    
    #creating the output image vector
    outputImg = np.zeros(m)
    
    #flip the vector of weights
    weights = np.flip(weights,0)
    
    #apply the filter
    for i in range(m):
        sumVector = 0
        for j in range(-a,a+1):
            if i+j>=0:
                if i+j>=m:
                    sumVector += vectorImg[(i+j)%m] * weights[j+a]
                else:
                    sumVector += vectorImg[i+j] * weights[j+a]
            else:
                sumVector += vectorImg[m+i+j] * weights[j+a]
        outputImg[i] = sumVector 

    return outputImg

#calculates the arbitrary filter with frequency domain
def arbitraryFilterFrequencyDomain(vectorImg, weights):
    
    #size of vector
    n = vectorImg.shape[0]

    #vector that will contain the output image
    F = np.zeros(n, dtype=np.complex64) 
    
    #vector that will contain the frequency
    W = np.zeros(n, dtype=np.complex64)

    #indices for x
    x = np.arange(n)
    
    #apply the fourier series for image
    for u in np.arange(n):
        F[u] = np.sum(vectorImg*np.exp((-1j*2*np.pi*u*x)/n))

    #reshape the vector
    tempVector = np.zeros(n)
    for i in range(weights.shape[0]):
        tempVector[i] = weights[i]
    weights = tempVector
    
    #apply the fourier series for weights
    for u in np.arange(n):
        W[u] = np.sum(weights*np.exp((-1j*2*np.pi*u*x)/n))
    
    #calculating the inverse
    R = np.multiply(F,W)
    outputImg = np.zeros(n, dtype=np.complex64) 
    for u in np.arange(1,n):
        outputImg[u] = np.sum(R*np.exp((1j*2*np.pi*x*u)/n))/n
    
    outputImg = np.real(outputImg)
    return outputImg


#calculates the gaussian filter with spatial domain
def gaussianFilterSpaceDomain(vectorImg,sizeFilter,standardDeviation):

    #size of vector
    m = vectorImg.shape[0]
    
    #this is used to perform the convolution
    a = int((sizeFilter-1)/2)
    
    #creating the output image vector
    outputImg = np.zeros(m)
    
    #vector of weights that will be used in filter
    weights = np.zeros(sizeFilter)
    
    #create filter
    for i in range(-a,a+1):
        weights[i+a] = np.exp(-math.pow((i/standardDeviation),2)/2)/(math.sqrt(2*math.pi)*standardDeviation)
        
    #flip the vector of weights
    weights = np.flip(weights,0)
    
    #normalize the weights
    weights = weights/np.sum(weights)

    #apply the filter
    for i in range(m):
        sumVector = 0
        for j in range(-a,a+1):
            if i+j>=0:
                if i+j>=m:
                    sumVector +=  vectorImg[(i+j)%m] * weights[j+a]
                else:
                    sumVector +=  vectorImg[i+j] * weights[j+a]
            else:
                sumVector += vectorImg[m+i+j] * weights[j+a]
        outputImg[i] = sumVector
    
    return outputImg

#calculates the gaussian filter with frequency domain
def gaussianFilterFrequencyDomain(vectorImg,sizeFilter,standardDeviation):
    
    #size of vector
    m = vectorImg.shape[0]
    
    #this is used to perform the convolution
    a = int((sizeFilter-1)/2)
    
    #creating the output image vector
    outputImg = np.zeros(m)
    
    #vector of weights that will be used in filter
    weights = np.zeros(sizeFilter)
    
    #create filter
    for i in range(-a,a+1):
        weights[i+a] = np.exp(-math.pow((i/standardDeviation),2)/2)/(math.sqrt(2*math.pi)*standardDeviation)
        

    #flip the vector of weights
    weights = np.flip(weights,0)
   
    #normalize the weights
    weights = weights/np.sum(weights)
    
    #vector that will contain the output image
    F = np.zeros(m, dtype=np.complex64) 
    
    #vector that will contain the frequency
    W = np.zeros(m, dtype=np.complex64)

    #indices for x
    x = np.arange(m)
    
    #apply the fourier series for image
    for u in np.arange(m):
        F[u] = np.sum(vectorImg*np.exp((-1j*2*np.pi*u*x)/m))

    #reshape the vector
    tempVector = np.zeros(m)
    for i in range(weights.shape[0]):
        tempVector[i] = weights[i]
    weights = tempVector
    
    #apply the fourier series for weights
    for u in np.arange(m):
        W[u] = np.sum(weights*np.exp((-1j*2*np.pi*u*x)/m))
    
    #calculating the inverse
    R = np.multiply(F,W)
    outputImg = np.zeros(m, dtype=np.complex64) 
    for u in np.arange(m):
        outputImg[u] = np.sum(R*np.exp((1j*2*np.pi*x*u)/m))/m
    
    outputImg = np.real(outputImg)
    return outputImg


#calculating the square error
def error(vectorImg,outputImg):
    
    #normalizing
    maxO = outputImg.max()
    minO = outputImg.min()
    outputImg = ((outputImg-minO)/(maxO-minO))*255
    
    #convert to int
    outputImg = np.uint(outputImg)

    #size of vector
    m = vectorImg.shape[0]

    sumError = 0
    for i in range(m):
            sumError += math.pow((vectorImg[i] - outputImg[i]),2)
    
    sumError = math.sqrt((sumError/m))
    print(sumError)

#Read the input data
def inputData():
    
    #name of image
    nameImage = str(input()).rstrip()

    #choice the filter
    typeFilter = int(input())

    #size of filter
    sizeFilter = int(input())

    #weights of filter
    if typeFilter == 1:
        w = str(input())
        listW = w.split()
        weights = np.zeros(sizeFilter)
        for i in range(sizeFilter):
            weights[i] = float(listW[i])
    else:
        standardDeviation = float(input())

    #read the filtering domain
    filteringDomain = int(input())

    #read image
    matrixImg = img.imread(nameImage)

    #transforming matrix into vector
    vectorImg = np.zeros(matrixImg.shape[0]*matrixImg.shape[1])
    for i in range(0,matrixImg.shape[0]):
        for j in range(0,matrixImg.shape[1]):
            vectorImg[(i*matrixImg.shape[1])+j] = matrixImg[i][j]            
    
    #applies the filter
    if typeFilter == 1:
        if filteringDomain == 1:
            outputImg = arbitraryFilterSpatialDomain(vectorImg,weights)
        else:
            outputImg = arbitraryFilterFrequencyDomain(vectorImg,weights)
    else:
        if filteringDomain ==1:
            outputImg = gaussianFilterSpaceDomain(vectorImg,sizeFilter,standardDeviation)
        else:
            outputImg = gaussianFilterFrequencyDomain(vectorImg,sizeFilter,standardDeviation)
    
    #calculates the erro
    error(vectorImg,outputImg)

#call for input data
inputData();
