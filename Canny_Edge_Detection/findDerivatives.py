'''
  File name: findDerivatives.py
  Author:Weiyi Tang
  Date created: 09/15/2019
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''
import utils
import numpy as np
from scipy import signal

def findDerivatives(I_gray):
  # TODO: your code here
  mu=0
  sigma=2
  G=utils.GaussianPDF_2D(mu, sigma, 3,3)#form 2d gussian
  dx,dy=np.gradient(G, axis=(1,0))#form 2d gussian derivative in x and y direction
  
  Magx=signal.convolve2d(I_gray,dx,'same')
  Magy=signal.convolve2d(I_gray,dy,'same')
  
  Mag=np.sqrt(Magx*Magx+Magy*Magy)
  Ori=np.arctan2(Magy,Magx,out=None)

  return [Mag,Magx,Magy,Ori]
  
  
  
  
