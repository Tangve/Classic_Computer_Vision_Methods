# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:00:03 2019

@author: TANG VE
"""
from genEngMap import genEngMap
from cumMinEngVer import cumMinEngVer
import numpy as np
import cv2
import matplotlib.pyplot as plt

# left side of image carving
# w1 is the distance from left side of the image
def usingseam_l(w1,img):
    print('left carving')
    img_p=img[:,0:w1]
    e=genEngMap(img_p)
    Mx,Tbx=cumMinEngVer(e)
    h,w=Mx.shape
    E=min(Mx[h-1])
    index=[]
    for i in range(w):
        if Mx[h-1][i]==E:
            index.append([h-1,i])
            break
    temp=index[0]
    for i in range(h-1,0,-1):
        if Tbx[i][temp[1]]==-1:
            temp=[i-1,temp[1]-1]
            index.append(temp)
        if Tbx[i][temp[1]]==0:
            temp=[i-1,temp[1]] 
            index.append(temp)
        if Tbx[i][temp[1]]==1:
            temp=[i-1,temp[1]+1]
            index.append(temp)
    index.sort()
    mask=np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if j>index[i][1]:
                mask[i][j]=1
    I_p=np.copy(img_p)
    I_p[:,:,0]=I_p[:,:,0]*mask
    I_p[:,:,1]=I_p[:,:,1]*mask
    I_p[:,:,2]=I_p[:,:,2]*mask
    I=np.copy(img)
    I[:,0:w1]=I_p
    
    return I

# right side of image carving
# w1 is the distance from left side of the image
def usingseam_r(w1,img):
    print('right carving')
    img_p=img[:,w1:]
    e=genEngMap(img_p)
    Mx,Tbx=cumMinEngVer(e)
    h,w=Mx.shape
    E=min(Mx[h-1])
    index=[]
    for i in range(w):
        if Mx[h-1][i]==E:
            index.append([h-1,i])
            break
    temp=index[0]
    for i in range(h-1,0,-1):
        if Tbx[i][temp[1]]==-1:
            temp=[i-1,temp[1]-1]
            index.append(temp)
        if Tbx[i][temp[1]]==0:
            temp=[i-1,temp[1]] 
            index.append(temp)
        if Tbx[i][temp[1]]==1:
            temp=[i-1,temp[1]+1]
            index.append(temp)
    index.sort()
    mask=np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if j<index[i][1]:
                mask[i][j]=1
    I_p=np.copy(img_p)
    I_p[:,:,0]=I_p[:,:,0]*mask
    I_p[:,:,1]=I_p[:,:,1]*mask
    I_p[:,:,2]=I_p[:,:,2]*mask
    I=np.copy(img)
    I[:,w1:]=I_p
    
    return I


if __name__ == "__main__":
    img = cv2.imread('middle1.jpg')
    index=usingseam_r(250,img)
    plt.imshow(index)
    plt.show()
    
