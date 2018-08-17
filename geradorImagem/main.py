# -*- coding: utf-8 -*-

import numpy as np
import math
import random

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
Title: Gerador de Imagens
'''

#Generating scene 1 function
def generatingFunction1(sceneImage,C):
    for i in range(0,C):
        for j in range(0,C):
            sceneImage[i][j] = float(i+j)

#Generating scene 2 function
def generatingFunction2(sceneImage,C,Q):
    for i in range(0,C):
        for j in range(0,C):
            sceneImage[i][j] = float(abs(math.sin(i/Q) + math.sin(j/Q)))

#Generating scene 3 function
def generatingFunction3(sceneImage,C,Q):
    for i in range(0,C):
        for j in  range(0,C):
            sceneImage[i][j] = float(abs((i/Q) - math.sqrt(j/Q)))

#Generating scene 4 function
def generatingFunction4(sceneImage,C):
    for i in range(0,C):
        for j in range(0,C):
            sceneImage[i][j] = float(random.random())

#Generating scene 5 function
def generatingFunction5(sceneImage,C):
    
    x = y = 0
    sceneImage[x][y] = float(1)
    
    for i in  range(0,int((C*C)/2)):
        dx = random.randint(-1,1) 
        dy = random.randint(-1,1) 
        x = (x + dx)%C
        sceneImage[x][y] = float(1)
        y = (y + dy)%C
        sceneImage[x][y] = float(1)

#Normalizing the scene image
def normalizing(sceneImage,C):
    
    #Get the min and max value to perform the normalization
    min = sceneImage.min()
    max = sceneImage.max()
    
    #Doing the normalization
    sceneImage = ((sceneImage-min)/(max-min))*65535

#Creating a digital image
def scanning(sceneImage,digitalImage,C,N,B):
    
    #displacement calculation
    d = int(C/N)
   
    for i in range(0,N):
        for j in range(0,N):
            #calculating the local maximum
            if(d > 1):
                digitalImage[i][j] = sceneImage[i*d:(i*d)+d,j*d:(j*d)+d].max()
            else:
                digitalImage[i][j] = sceneImage[i][j]
    
    #Convert numbers to the range 0-255
    globalMax = digitalImage.max()
    digitalImage = (digitalImage/globalMax)*255

    #converting image to uint8
    digitalImage = np.uint8(digitalImage)
    
    #shift
    digitalImage = digitalImage >> (8-B)

    return digitalImage

#Calculating the error between two images
def error(digitalImage,R,N):
    
    #euclidean distance
    errorSum = 0.0
    for i in  range(0,N):
        for j in range(0,N):
                errorSum += pow((int(digitalImage[i][j]) - int(R[i][j])),2)
    errorSum = math.sqrt(errorSum)
    
    #print error
    print(errorSum)

#Reading data
def inputData():

    #Name of image
    filename = str(input()).rstrip()
    R = np.load(filename)
    R = np.uint8(R)
    
    #Side image size of the scene
    C = int(input())

    #Type of function that will be performed
    typeFunction = int(input())

    #Parameter Q that will be used in function
    Q = int(input())

    #Digital image side size
    N = int(input())

    #Number of bits used in an image
    bits = int(input())

    #Seed used in functions
    seed = int(input())

    #Definition of seed
    random.seed(seed)

    #Creating a matrix that will be used for a scene image
    sceneImage = np.zeros((C,C))
    
    #Generating the scene image
    if typeFunction == 1:
        generatingFunction1(sceneImage,C)
    elif typeFunction == 2:
        generatingFunction2(sceneImage,C,Q)
    elif typeFunction == 3:
     generatingFunction3(sceneImage,C,Q)
    elif typeFunction == 4:
     generatingFunction4(sceneImage,C)
    elif typeFunction == 5:
        generatingFunction5(sceneImage,C)
    
    #Normalizing
    normalizing(sceneImage,C)

    #Create a matrix for digital image
    digitalImage = np.zeros((N,N))

    #Generating the digital image
    digitalImage = scanning(sceneImage,digitalImage,C,N,bits)

    #Calculating the error
    error(digitalImage,R,N)

#call the start function
inputData()
