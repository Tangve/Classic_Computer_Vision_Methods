'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
from est_homography import est_homography
import numpy as np
from random import sample

def ransac_est_homography(x1, y1, x2, y2, thresh):
    Max=0
    N=len(x1)
    for i in range(1000):
        index=sample(range(N), 4)
        x=x1[index]
        y=y1[index]
        X=x2[index]
        Y=y2[index]
        h=est_homography(x, y, X, Y)
        u=np.concatenate((x1.reshape(1,N), y1.reshape(1,N), np.ones((1, N))), axis=0)
        v=np.dot(h, u)
        v=v[:2]/v[2,:]
        v_r=np.concatenate((x2.reshape(1,N), y2.reshape(1,N)), axis=0)
        error=np.sqrt(np.sum((v_r - v)**2, axis=0))
        inlier_in = np.argwhere(error < thresh)
        num=error[error < thresh].size
        if num>Max:
            Max=num
            inlier_index=inlier_in
            H=h
    return H, inlier_index