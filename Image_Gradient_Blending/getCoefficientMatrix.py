'''
  File name: getCoefficientMatrix.py
  Author:
  Date created:
'''

import numpy as np

def getCoefficientMatrix(indexes):
    indexes=np.int32(indexes)
    N=int(indexes.max())
    h,w=indexes.shape
    count=1
    A=np.int32(np.ones([N,N]))
    for i in range(h):
        for j in range(w):
            if indexes[i][j]==count:
                a=np.zeros([1,N])
                a[0][count-1]=4
                if j-1>0 and indexes[i][j-1]>0:
                    a[0][indexes[i][j-1]-1]=-1
                if j+1<w and indexes[i][j+1]>0:
                    a[0][indexes[i][j+1]-1]=-1 
                if i-1>0 and indexes[i-1][j]>0:
                    a[0][indexes[i-1][j]-1]=-1
                if i+1<h and indexes[i+1][j]>0:
                    a[0][indexes[i+1][j]-1]=-1
                A[count-1]=a
                count+=1
    return A