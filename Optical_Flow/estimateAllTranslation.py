
import cv2
import numpy as np
from findDerivatives import findDerivatives
from estimateFeatureTranslation import estimateFeatureTranslation


def estimateAllTranslation(startXs, startYs, img1, img2):
    newXs = np.empty_like(startXs)
    newYs = np.empty_like(startYs)
    for i, (X,Y) in enumerate(zip(startXs.T, startYs.T)):
        for j, (startX, startY) in enumerate(zip(X, Y)):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            Ix, Iy = findDerivatives(gray1)
            if startX >= 0 and startY >= 0:
                newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, gray1, gray2)
                newXs[j][i] = newX
                newYs[j][i] = newY
            else:
                newXs[j][i] = -1
                newYs[j][i] = -1
    return newXs, newYs
