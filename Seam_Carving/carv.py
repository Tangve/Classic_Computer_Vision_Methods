'''
  File name: carv.py
  Author:
  Date created:
'''

'''
  File clarification:
    Aimed to handle finding seams of minimum energy, and seam removal, the algorithm
    shall tackle resizing images when it may be required to remove more than one seam, 
    sequentially and potentially along different directions.
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT nr: the numbers of rows to be removed from the image.
    - INPUT nc: the numbers of columns to be removed from the image.
    - OUTPUT Ic: (n − nr) × (m − nc) × 3 matrix representing the carved image.
    - OUTPUT T: (nr + 1) × (nc + 1) matrix representing the transport map.
'''
import numpy as np
from rmVerSeam import rmVerSeam
from rmHorSeam import rmHorSeam
from genEngMap import genEngMap
from cumMinEngHor import cumMinEngHor
from cumMinEngVer import cumMinEngVer
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio

def rr(I):
    e=genEngMap(I)
    My,Tby=cumMinEngHor(e)
    Iy,E=rmHorSeam(I, My, Tby)
    return Iy,E


def rc(I):
    e=genEngMap(I)
    Mx,Tbx=cumMinEngVer(e)
    Ix,E=rmVerSeam(I, Mx, Tbx)
    return Ix,E

def carv(I, nr, nc): 
    h,w=I[:,:,0].shape
    T=np.zeros([nr+1,nc+1])
    Tp=np.ndarray(shape=(nr+1,nc+1),dtype=object)
    dire=np.zeros([nr+1,nc+1])
    for i in range(nr+1):
        for j in range(nc+1):
            if i==j==0:
                T[i][j]=0
                Tp[i][j]=I
            elif i==0 and j!=0:
                Ix,E=rc(Tp[i][j-1])
                T[i][j]=T[i][j-1]+E
                Tp[i][j]=Ix
                dire[i][j]=-1
            elif i!=0 and j==0:
                Iy,E=rr(Tp[i-1][j])
                T[i][j]=T[i-1][j]+E
                Tp[i][j]=Iy
                dire[i][j]=1
            else:
                Ix,E1=rc(Tp[i][j-1])
                Iy,E2=rr(Tp[i-1][j])
                Min=min(E1+T[i][j-1],E2+T[i-1][j])
                T[i][j]=Min
                if E1+T[i][j-1]==Min:
                    Tp[i][j]=Ix
                    dire[i][j]=-1
                if E2+T[i-1][j]==Min:
                    Tp[i][j]=Iy
                    dire[i][j]=1
    result=np.zeros([nr+nc+1,h,w,3])
    ht,wt=Tp[nr][nc][:,:,0].shape
    Temp=np.zeros([h,w,3])
    tempT00=np.concatenate((Tp[nr][nc][:,:,0],np.zeros([h-ht,wt])),axis=0)
    tempT10=np.concatenate((Tp[nr][nc][:,:,1],np.zeros([h-ht,wt])),axis=0)
    tempT20=np.concatenate((Tp[nr][nc][:,:,2],np.zeros([h-ht,wt])),axis=0)
    tempT01=np.concatenate((tempT00,np.zeros([h,w-wt])),axis=1)
    tempT11=np.concatenate((tempT10,np.zeros([h,w-wt])),axis=1)
    tempT21=np.concatenate((tempT20,np.zeros([h,w-wt])),axis=1)
    Temp[:,:,0]=tempT01
    Temp[:,:,1]=tempT11
    Temp[:,:,2]=tempT21
    result[nr+nc]=Temp
    index=[nr,nc]
    for i in range(nr+nc):
        if dire[index[0]][index[1]]==1:
            temp=Tp[index[0]-1][index[1]]
            h1,w1=temp[:,:,0].shape
            temp2=np.zeros([h,w,3])
            tempR=np.concatenate((temp[:,:,0],np.zeros([h-h1,w1])),axis=0)
            tempR=np.concatenate((tempR,np.zeros([h,w-w1])),axis=1)
            tempG=np.concatenate((temp[:,:,1],np.zeros([h-h1,w1])),axis=0)
            tempG=np.concatenate((tempG,np.zeros([h,w-w1])),axis=1)
            tempB=np.concatenate((temp[:,:,2],np.zeros([h-h1,w1])),axis=0)
            tempB=np.concatenate((tempB,np.zeros([h,w-w1])),axis=1)
            temp2[:,:,0]=tempR
            temp2[:,:,1]=tempG
            temp2[:,:,2]=tempB
            result[nr+nc-i-1]=temp2
            index=[index[0]-1,index[1]]
        if dire[index[0]][index[1]]==-1:
            temp=Tp[index[0]][index[1]-1]
            h1,w1=temp[:,:,0].shape
            temp2=np.zeros([h,w,3])
            tempR=np.concatenate((temp[:,:,0],np.zeros([h-h1,w1])),axis=0)
            tempR=np.concatenate((tempR,np.zeros([h,w-w1])),axis=1)
            tempG=np.concatenate((temp[:,:,1],np.zeros([h-h1,w1])),axis=0)
            tempG=np.concatenate((tempG,np.zeros([h,w-w1])),axis=1)
            tempB=np.concatenate((temp[:,:,2],np.zeros([h-h1,w1])),axis=0)
            tempB=np.concatenate((tempB,np.zeros([h,w-w1])),axis=1)
            temp2[:,:,0]=tempR
            temp2[:,:,1]=tempG
            temp2[:,:,2]=tempB
            result[nr+nc-i-1]=temp2
            index=[index[0],index[1]-1]
    Ic=Tp[nr][nc]
    return Ic, T, result

if __name__ == "__main__":
    I=plt.imread('source.png')
    nr=20
    nc=20
    Ic,T,result=carv(I, nr, nc)
    
    frames, x, y, z = result.shape
    morph_list = []
    for i in range(0, frames):
        image = Image.fromarray(result[i, :, :, :], 'RGB')
        morph_list.append(result[i, :, :, :])
        image.save('photo.jpg')
        image.show()
        plt.show()
        imageio.mimsave('./Results.gif', morph_list)
    #Id=Ic/255
    plt.imshow(Ic)
    plt.savefig("resultimg.png")
    plt.show()