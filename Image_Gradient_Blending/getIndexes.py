'''
  File name: getIndexes.py
  Author:
  Date created:
'''


import numpy as np

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
    h,w=mask.shape
    temp=np.column_stack((np.zeros([h,offsetX]),mask,np.zeros([h,targetW-w-offsetX])))
    mask1=np.row_stack((np.zeros([offsetY,targetW]),temp,np.zeros([targetH-h-offsetY,targetW])))
    indexes=np.zeros([targetH,targetW])
    count=1
    for i in range(targetH):
        for j in range(targetW):
            if mask1[i][j]==1:
                indexes[i][j]=count
                count+=1
    return indexes