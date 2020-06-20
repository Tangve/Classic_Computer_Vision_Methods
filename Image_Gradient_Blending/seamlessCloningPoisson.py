'''
  File name: seamlessCloningPoisson.py
  Author:
  Date created:
'''
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt


from getIndexes import getIndexes
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg


def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
    
    targetH,targetW,L=targetImg.shape
    
    indexes=getIndexes(mask, targetH, targetW, offsetX, offsetY)
    
    A=getCoefficientMatrix(indexes)
    
    source1=np.copy(sourceImg[:,:,0])
    source2=np.copy(sourceImg[:,:,1])
    source3=np.copy(sourceImg[:,:,2])
    
    target1=np.copy(targetImg[:,:,0])
    target2=np.copy(targetImg[:,:,1])
    target3=np.copy(targetImg[:,:,2])
    
    bred=getSolutionVect(indexes, source1, target1, offsetX, offsetY)
    bgreen=getSolutionVect(indexes, source2, target2, offsetX, offsetY)
    bblue=getSolutionVect(indexes, source3, target3, offsetX, offsetY)

    red=np.linalg.solve(A,bred)
    green=np.linalg.solve(A,bgreen)
    blue=np.linalg.solve(A,bblue)
    
    '''
    red=np.clip(red,0,1).copy()
    green=np.clip(bgreen,0,1).copy()
    blue=np.clip(blue,0,1).copy()
    '''
    
    
    Red=red.T
    Green=green.T
    Blue=blue.T
    
    resultImg=reconstructImg(indexes, Red, Green, Blue, targetImg)
    

    plt.figure(figsize=(12,12))
    
    plt.imshow(resultImg)
    
    plt.show()
    
    return resultImg



if __name__ == "__main__":
    sourceImg=plt.imread('1_source.jpg')
    targetImg=plt.imread('1_background.jpg')
    targetImg=targetImg[:,:,0:3].copy()
    targetImg=targetImg/targetImg.max()
    sourceImg=sourceImg/sourceImg.max()
    mask=plt.imread('1_mask.png')
    try:
        mask=mask[:,:,0]
    except:
        mask=mask
    h,w=mask.shape
    H,W,L=targetImg.shape
    
    offsetX=200
    offsetY=200
    '''
    offsetX=int(0.5*(W-w))
    offsetY=int(0.5*(H-h))
    '''
    
    resultImg=seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY)