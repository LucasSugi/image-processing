# -*- coding: utf-8 -*-

import numpy as np
import imageio as img

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Realce e Superresolução 
'''

#Apply the arbitrary filter
def arbitraryFilter(image,weights):

    #apply the discrete Fourier Transform in image
    F = np.fft.fft2(image)
    
    #temporary matrix that will be used to shape weights equal to image
    tempMatrix = np.zeros([image.shape[0],image.shape[1]]) 
    
    #copy weights to the tempMatrix
    M,N = weights.shape
    for i in range(M):
        for j in range(N):
            tempMatrix[i][j] = weights[i][j]

    #apply the discrete Fourier Transform in weights
    W = np.fft.fft2(tempMatrix)
    
    #multiply point by point
    return np.multiply(F,W)

#apply the gaussian function
def gaussian(x,y,sD):

    return (-1/(np.pi*np.power(sD,4))) * (1-((np.power(x,2)+np.power(y,2))/(2*np.power(sD,2))))* (np.exp((-np.power(x,2)-np.power(y,2))/(2*np.power(sD,2)))) 

#Apply the Gaussian Laplacian filter
def gaussianLaplacianFilter(image,n, standardDeviation):
    
    #verify the linearity in the matrix of weights
    linearity = int((n-1)/2)
    linearity = float(5/linearity)

    #populating the matrix off weights
    pos = 0
    neg = 0
    weights = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            weights[i][j] = gaussian(-5 + (linearity*j),5 - (linearity*i),standardDeviation)
            if(weights[i][j] > 0):
                pos += weights[i][j]
            else:
                neg += weights[i][j]
    
    #normalizing the matrix
    for i in range(n):
        for j in range(n):
            if (weights[i][j] < 0):
                weights[i][j] = weights[i][j] * (-pos/neg)

    #temporary matrix that will be used to shape weights equal to image
    tempMatrix = np.zeros([image.shape[0],image.shape[1]]) 
    
    #copy weights to the tempMatrix
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            tempMatrix[i][j] = weights[i][j]
    
    #apply the discrete Fourier Transform in weights
    W = np.fft.fft2(tempMatrix)

    #apply the discrete Fourier Transform in image
    F = np.fft.fft2(image)

    #multiply point by point
    return np.multiply(F,W)

#Apply the convolution
def convolution(image, w, x, y):

    #extract submatrix
    sub = image[x-1:x+2,y-1:y+2]
        
    #multiply point by point
    return np.sum(np.multiply(sub,w))

#Apply the sobel operator
def sobelOperator(image):
    
    #creating the 2 filters
    Fx = np.zeros([3,3])
    Fx[0][0] = 1
    Fx[1][0] = 2
    Fx[2][0] = 1
    Fx[0][2] = -1
    Fx[1][2] = -2
    Fx[2][2] = -1
    Fy = Fx.transpose()
    
    #image with edge
    imageEdge = np.pad(image,(1,1),'constant',constant_values=0)
        
    #convolution
    N,M = image.shape
    Ix = np.zeros([N,M])
    Iy = np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            Ix[i][j] = convolution(imageEdge,Fx,i+1,j+1)
            Iy[i][j] = convolution(imageEdge,Fy,i+1,j+1)

    #generating the Iout
    Iout = np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            Iout[i][j] = np.sqrt(np.power(Ix[i][j],2) + np.power(Iy[i][j],2))

    return np.fft.fft2(Iout)

#rate the image
def knn(Vout,dataset,labels):
    
    #load file with images
    data = np.load(dataset)
    
    #load file with labels
    dataLabels = np.load(labels)

    #knn
    M = data.shape[0]
    distances = np.zeros(M) 
    for i in range(M):
        distances[i] = np.linalg.norm(Vout-data[i])
     
    #load file with labels
    closer = np.argmin(distances)
    print(dataLabels[closer])
    print(closer)

#read the data input
def inputData():

    #name of image
    nameImage = str(input()).rstrip()

    #choice of method that we will appli in image
    typeMethod = int(input())
    
    #verify what is the method for read the right values
    if typeMethod == 1:
        readValue = str(input()).rstrip()
        h,w = readValue.split()
        h = int(h)
        w = int(w)
        weights = np.zeros([h,w])
        for i in range(h):
                readValue = str(input()).rstrip()
                weights[i] = readValue.split()
    elif typeMethod == 2: 
        n = int(input())
        standardDeviation = float(input())
    
    #numbers that represent the cut's in image
    readValue = str(input()).rstrip()
    Hlb,Hub,Wlb,Wub = readValue.split()
    Hlb = float(Hlb)
    Hub = float(Hub)
    Wlb = float(Wlb)
    Wub = float(Wub)

    #name of file dataset
    dataset = str(input()).rstrip()

    #name of labels of dataset
    labels = str(input()).rstrip()
    
    #read the image
    image = img.imread(nameImage)

    #verify what is the method of extraction features
    if typeMethod == 1:
        Iout = arbitraryFilter(image,weights)
    elif typeMethod == 2:
        Iout = gaussianLaplacianFilter(image,n,standardDeviation)
    else:
        Iout = sobelOperator(image)    

    #apply the cut in image
    H,W = Iout.shape
    H = int(H/2)
    W = int(W/2)
    Icut1 = Iout[0:H,0:W]
    
    #second cut
    H,W = Icut1.shape
    Hlb = int(Hlb * H)
    Hub = int(Hub * H)
    Wlb = int(Wlb * W)
    Wub = int(Wub * W)
    Icut2 = Icut1[Hlb:Hub,Wlb:Wub]
    
    #apply the knn
    knn(np.asarray(Icut2).reshape(-1),dataset,labels)
    
#Call for function that will be read the data
inputData();
