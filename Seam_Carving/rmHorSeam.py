'''
  File name: rmHorSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes horizontal seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - INPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
    - OUTPUT Iy: (n − 1) × m × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
from rmVerSeam import rmVerSeam
import numpy as np

def rmHorSeam(I, My, Tby):
    I1=np.copy(I)
    h,w=I1[:,:,0].shape
    I2=np.zeros([w,h,3])
    I2[:,:,0]=I1[:,:,0].T
    I2[:,:,1]=I1[:,:,1].T
    I2[:,:,2]=I1[:,:,2].T
    My1=My.T
    Tby1=Tby.T
    IyT1,E=rmVerSeam(I2, My1, Tby1)
    h,w=IyT1[:,:,0].shape
    Iy=np.zeros([w,h,3])
    Iy[:,:,0]=IyT1[:,:,0].T
    Iy[:,:,1]=IyT1[:,:,1].T
    Iy[:,:,2]=IyT1[:,:,2].T
    return Iy, E
