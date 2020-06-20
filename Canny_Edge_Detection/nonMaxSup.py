'''
  File name: nonMaxSup.py
  Author:Weiyi Tang
  Date created: 09/15/2019
'''

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''
import numpy as np
from interp import interp2

def nonMaxSup(Mag, Ori):
    
    h,w=Mag.shape
    
    lx=np.cos(Ori)
    ly=np.sin(Ori)
    
    x,y=np.meshgrid(np.arange(w),np.arange(h))
    xq_p=x+lx
    yq_p=y-ly
    xq_n=x-lx
    yq_n=y+ly
    
    Pos=interp2(Mag, xq_p, yq_p)
    Neg=interp2(Mag, xq_n, yq_n)
    
    M=np.int64(Mag-Pos>0)*np.int64(Mag-Neg>0)
    
    return M
    
    
  