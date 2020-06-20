import numpy as np
from scipy import signal

def findDerivatives(img):
    dx = np.array([1, -1]).reshape(1, -1)
    dy = np.array([1, -1]).reshape(-1, 1)
    Ix = signal.convolve2d(img, dx)
    Iy = signal.convolve2d(img, dy)
    return Ix, Iy
