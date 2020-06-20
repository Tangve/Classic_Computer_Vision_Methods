'''
  File name: cumMinEngVer.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the vertical seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - OUTPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
'''
import numpy as np

def cumMinEngVer(e):
    h,w=e.shape
    Mx=np.zeros([h,w])
    Tbx=np.zeros([h,w])
    Mx[0]=e[0]
    for i in range(1,h):
        for j in range(w):
            if j==0:
                Min=min(Mx[i-1][j],Mx[i-1][j+1])
                Mx[i][j]=Min+e[i][j]
                if Mx[i-1][j]==Min:
                    Tbx[i][j]=0
                else:
                    Tbx[i][j]=1
            elif j==w-1:
                Min=min(Mx[i-1][j-1],Mx[i-1][j])
                Mx[i][j]=Min+e[i][j]
                if Mx[i-1][j-1]==Min:
                    Tbx[i][j]=-1
                else:
                    Tbx[i][j]=0
            else:
                Min=min(Mx[i-1][j-1],Mx[i-1][j],Mx[i-1][j+1])
                Mx[i][j]=Min+e[i][j]
                if Mx[i-1][j-1]==Min:
                    Tbx[i][j]=-1
                elif Mx[i-1][j]==Min:
                    Tbx[i][j]=0
                else:
                    Tbx[i][j]=1         
    return Mx, Tbx