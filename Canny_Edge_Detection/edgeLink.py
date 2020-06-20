'''
  File name: edgeLink.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use hysteresis to link edges based on high and low magnitude thresholds
    - Input M: H x W logical map after non-max suppression
    - Input Mag: H x W matrix represents the magnitude of gradient
    - Input Ori: H x W matrix represents the orientation of gradient
    - Output E: H x W binary matrix represents the final canny edge detection map
'''
import numpy as np
from interp import interp2


def edgeLink(M, Mag, Ori):
    h,w=Mag.shape
    M1=M*Mag
    mean=M1.sum()/ M.sum() 
    #high=0.08
    #low=0.035
    high=1*mean
    low=0.2*mean
    
    High=high*np.ones([h,w])
    Low=low*np.ones([h,w])
    
    M=M*Mag
    hpoint=np.int64(M-High>0)
    mpoint=np.int64(High-M>0)*np.int64(M-Low>0)
    temp=mpoint
    E=hpoint
    
    lx_p=np.sin(Ori)
    ly_p=np.cos(Ori)
    x,y=np.meshgrid(np.arange(w),np.arange(h))
    xq_p=x+lx_p
    yq_p=y+ly_p
    xq_n=x-lx_p
    yq_n=y-ly_p
    
    while(temp.sum()>0):
        Pos1=interp2(E*Mag, xq_p, yq_p)*mpoint
        Neg1=interp2(E*Mag, xq_n, yq_n)*mpoint
        temp=1*np.logical_or(np.int64(Pos1-High>0),np.int64(Neg1-High>0))
        E=E+temp
        mpoint=mpoint-temp
        
    
    return E

            

                