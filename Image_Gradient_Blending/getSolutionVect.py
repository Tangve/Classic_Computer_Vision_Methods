'''
  File name: getSolutionVect.py
  Author:
  Date created:
'''

import numpy as np
from scipy import signal



def getSolutionVect(indexes, source, target, offsetX, offsetY):
    indexes=np.int32(indexes).copy()
    h,w=source.shape
    Lap=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    Dsource=signal.convolve2d(source,Lap,'same')
    indexes1=indexes[offsetY:offsetY+h,offsetX:offsetX+w].copy()
    N=int(indexes.max())
    b=np.ones([N,1])
    for i in range(h):
        for j in range(w):
            if indexes1[i][j]!=0:
                temp=np.array([[Dsource[i][j]]])
                b[indexes1[i][j]-1]=temp
    H,W=target.shape
    for i in range(H):
        for j in range(W):
            if indexes[i][j]!=0:
                sum=0
                if indexes[i][j-1]==0:
                    sum+=target[i][j-1]
                if indexes[i][j+1]==0:
                    sum+=target[i][j+1]
                if indexes[i-1][j]==0:
                    sum+=target[i-1][j]
                if indexes[i+1][j]==0:
                    sum+=target[i+1][j]
                b[indexes[i][j]-1]+=sum
    return b
