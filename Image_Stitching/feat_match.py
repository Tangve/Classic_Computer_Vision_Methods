'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import cv2

def feat_match(des1, des2,kp1,kp2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)
    
    ratio_thresh = 0.6
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    new_kp1 = [i.queryIdx for i in good_matches]
    new_kp2 = [i.trainIdx for i in good_matches]
    x1 = [kp1[i].pt[0] for i in new_kp1]
    y1 = [kp1[j].pt[1] for j in new_kp1]
    x2 = [kp2[i].pt[0] for i in new_kp2]
    y2 = [kp2[j].pt[1] for j in new_kp2]
    return new_kp1,new_kp2,good_matches,x1,x2,y1,y2

