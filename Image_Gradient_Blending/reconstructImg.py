'''
  File name: reconstructImg.py
  Author:
  Date created:
'''
import numpy as np



def reconstructImg(indexes, red, green, blue, targetImg):
    h,w=indexes.shape
    indexR=np.copy(indexes)
    indexG=np.copy(indexes)
    indexB=np.copy(indexes)
    count=0
    for i in range(h):
        for j in range(w):
            if indexes[i][j]==count+1:
                indexR[i][j]=red[0][count]
                indexG[i][j]=green[0][count]
                indexB[i][j]=blue[0][count]
                count+=1
    comp1=np.ones([h,w])
    comp2=np.zeros([h,w])
    mask=np.int64(indexes>comp2)
    maskI=comp1-mask
    resultImg=np.copy(targetImg)
    layerR=targetImg[:,:,0].copy()*maskI+indexR
    layerG=targetImg[:,:,1].copy()*maskI+indexG
    layerB=targetImg[:,:,2].copy()*maskI+indexB
    resultImg[:,:,0]=np.copy(layerR)
    resultImg[:,:,1]=np.copy(layerG)
    resultImg[:,:,2]=np.copy(layerB)
    resultImg=np.copy(resultImg[:,:,0:3])
    return resultImg