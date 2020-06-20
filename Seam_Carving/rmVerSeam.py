'''
  File name: rmVerSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes vertical seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - INPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
    - OUTPUT Ix: n × (m - 1) × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
import numpy as np

def rmVerSeam(I, Mx, Tbx):
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
    I1=I.copy()
    for i in range(h):
        I1[:,:,0][i][index[i][1]:-1]=I1[:,:,0][i][index[i][1]+1:]
        I1[:,:,1][i][index[i][1]:-1]=I1[:,:,1][i][index[i][1]+1:]
        I1[:,:,2][i][index[i][1]:-1]=I1[:,:,2][i][index[i][1]+1:]
    I11=np.delete(I1[:,:,0],w-1,axis=1)
    I12=np.delete(I1[:,:,1],w-1,axis=1)
    I13=np.delete(I1[:,:,2],w-1,axis=1)
    Ix=np.zeros([h,w-1,3])
    Ix[:,:,0]=I11
    Ix[:,:,1]=I12
    Ix[:,:,2]=I13
    return Ix, E
