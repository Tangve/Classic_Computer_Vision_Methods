import cv2
#from scipy import signal
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    img1Copy = img1.copy()
    img2Copy = img2.copy()
    img1Func = RegularGridInterpolator((range(img1Copy.shape[0]), range(
        img1Copy.shape[1])), img1Copy, bounds_error=False, fill_value=None)
    img2Func = RegularGridInterpolator((range(img2Copy.shape[0]), range(
        img2Copy.shape[1])), img2Copy, bounds_error=False, fill_value=None)

    Ix_func = RegularGridInterpolator((range(Ix.shape[0]), range(
        Ix.shape[1])), Ix, bounds_error=False, fill_value=None)
    Iy_func = RegularGridInterpolator((range(Iy.shape[0]), range(
        Iy.shape[1])), Iy, bounds_error=False, fill_value=None)

    uv = np.zeros((2,1))
    newX = startX.copy()
    newY = startY.copy()
    iterates = 1
    while iterates <= 9:
        xi, yi = np.meshgrid(np.arange(startY-4.5, startY+4.6, 1),
                           np.arange(startX-4.5, startX+4.6, 1),)
        patch_indices = np.dstack((xi,yi))
        print(patch_indices.shape)
        print('after t times...')
        xt, yt = np.meshgrid(np.arange(newY-4.5, newY+4.6, 1),
                             np.arange(newX-4.5, newX+4.6, 1))
        patch_indices_t = np.dstack((xt,yt))
        print(patch_indices_t.shape)

        Ix_patch = Ix_func(patch_indices)
        Iy_patch = Iy_func(patch_indices)
        It_patch = img2Func(patch_indices_t) - img1Func(patch_indices)

        A = np.zeros((2, 2))
        A[0, 0] = np.sum(Ix_patch*Ix_patch)
        A[0, 1] = np.sum(Ix_patch*Iy_patch)
        A[1, 0] = np.sum(Ix_patch*Iy_patch)
        A[1, 1] = np.sum(Iy_patch*Iy_patch)
        b = np.zeros((2, 1))
        b[0, 0] = -np.sum(Ix_patch*It_patch)
        b[1, 0] = -np.sum(Iy_patch*It_patch)

        try:
            uv = np.linalg.solve(A,b)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                break
        newX = newX + uv[0, :]
        newY = newY + uv[1, :]
        if np.allclose(uv, np.array([[0], [0]]), rtol=0.05, atol=0.05):
            break
        iterates += 1
    return newX, newY