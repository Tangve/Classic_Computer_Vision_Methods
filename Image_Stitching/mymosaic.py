'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
import numpy as np
from corner_detector import corner_detector
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
import cv2
import matplotlib.pyplot as plt
from usingseam import usingseam_l
from usingseam import usingseam_r
from PIL import Image

    

def giveH_and_Plot(img1,img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # apply corner detector
    kp1 = corner_detector(gray1)
    kp2 = corner_detector(gray2)
    # apply feature descriptor
    kp1, des1 = feat_desc(gray1, kp1)
    kp2, des2 = feat_desc(gray2, kp2)
    # apply feature matches
    new_kp1, new_kp2, good_matches, x1, x2, y1, y2 = feat_match(
        des1, des2, kp1, kp2)

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # apply RANSAC homography
    H, index = ransac_est_homography(x1, y1, x2, y2, 0.8)

    img_keypoints1 = np.empty(
        (img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    img3 = cv2.drawKeypoints(img1, kp1, img_keypoints1)
    img_keypoints2 = np.empty(
        (img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
    img4 = cv2.drawKeypoints(img2, kp2, img_keypoints2)

    index = (index.T[0]).tolist()
    good2_matches = [good_matches[i] for i in index]

    img_matches = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    img5 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,
                           img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img1_matches = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    img6 = cv2.drawMatches(img1, kp1, img2, kp2, good2_matches,
                           img1_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # showing feature corner points
    plt.figure(1)
    plt.imshow(img3)
    plt.figure(2)
    plt.imshow(img4)
    # showing feature matching lines
    plt.figure(3)
    plt.imshow(img5)
    plt.figure(4)
    plt.imshow(img6)
    plt.show()

    # return homography computed at the end of RANSAC
    return H

# left mosaic padding 
def mosaic_l(image1, image2,seampic):
    homography=giveH_and_Plot(image1,image2)
    h = image2.shape[0]
    w = image2.shape[1]
    # get four corners of image2
    img2_Xcorners = np.array(
        [[0, 0, w-1, w-1], [0, h-1, 0, h-1], [1,   1,   1,   1]])
    # apply inverse homography matrix to get the coordinates of mosaic
    homography_inv = np.linalg.inv(homography)
    img2_Xcorners_trans = np.dot(homography_inv, img2_Xcorners)
    img2_Xcorners_trans = img2_Xcorners_trans / img2_Xcorners_trans[2]
    
    # calculate minX, minY, maxX and maxY for positive padding and negative padding
    min_x = np.floor(np.min(img2_Xcorners_trans[0])).astype(int)
    min_y = np.floor(np.min(img2_Xcorners_trans[1])).astype(int)
    max_x = np.ceil(np.max(img2_Xcorners_trans[0])).astype(int)
    max_y = np.ceil(np.max(img2_Xcorners_trans[1])).astype(int)
    
    neg_padding = -1 * min(np.array([0, min_y, min_x])).astype(int)
    pos_padding = max(np.array([max_y-h, max_x-w])).astype(int)
    
    num_colors = image1.shape[2]
    h1 = image1.shape[0]
    w1 = image1.shape[1]
    # create final mosaic image
    mosaic = np.zeros((neg_padding + h1 + pos_padding, neg_padding + w1 + pos_padding, num_colors)).astype(np.uint8)
    # fill the final mosaic image with the reference image(middle)
    mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = image1
    
    # calculate translation homography matrix
    translation = np.array([[1, 0, neg_padding],[0, 1, neg_padding],[0, 0, 1],])
    homography_trans = np.dot(translation, homography_inv)
    
    # transform image2 to mosiac image coordinate
    print("Warping left image...")
    warped_image = np.zeros(mosaic.shape).astype(np.uint8)
    warped_image = cv2.warpPerspective(image2, homography_trans, (warped_image.shape[1], warped_image.shape[0]))
    
    # pad left part with carved image
    mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = seampic
    
    nonzero_img1_inWarped = (np.sum(warped_image, axis=2) > 0)
    nonzero_img2_inMosaic = (np.sum(mosaic, axis=2) > 0)
    
    print("Combining two images...")
    # padding carved and original images to the mosiac final image
    for r in range(mosaic.shape[0]):
        for c in range(mosaic.shape[1]):
            if nonzero_img2_inMosaic[r, c] and nonzero_img1_inWarped[r, c]:
                mosaic[r,c,:] = mosaic[r,c,:]
            elif nonzero_img1_inWarped[r, c]:
                mosaic[r,c,:] = warped_image[r,c,:]
    mosaic1 = np.zeros((neg_padding + h1 + pos_padding, neg_padding + w1 + pos_padding, num_colors)).astype(np.uint8)
    mosaic1[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = image1
    
    nonzero_img1_inWarped1 = (np.sum(warped_image, axis=2) > 0)
    nonzero_img2_inMosaic1 = (np.sum(mosaic1, axis=2) > 0)
    
    for r in range(mosaic1.shape[0]):
        for c in range(mosaic1.shape[1]):
            if nonzero_img2_inMosaic1[r, c] and nonzero_img1_inWarped1[r, c]:
                mosaic1[r,c,:] = mosaic1[r,c,:]
            elif nonzero_img1_inWarped1[r, c]:
                mosaic1[r,c,:] = warped_image[r,c,:]
    return mosaic,mosaic1

# right mosaic padding
def mosaic_r(image1, image2,result):
    homography=giveH_and_Plot(image1,image2)
    h = image2.shape[0]
    w = image2.shape[1]
    # get four corners of image2
    img2_Xcorners = np.array(
        [[0, 0, w-1, w-1], [0, h-1, 0, h-1], [1,   1,   1,   1]])
    # apply inverse homography matrix to get the coordinates of mosaic
    homography_inv = np.linalg.inv(homography)
    img2_Xcorners_trans = np.dot(homography_inv, img2_Xcorners)
    img2_Xcorners_trans = img2_Xcorners_trans / img2_Xcorners_trans[2]
    
    # calculate minX, minY, maxX and maxY for positive padding and negative padding
    min_x = np.floor(np.min(img2_Xcorners_trans[0])).astype(int)
    min_y = np.floor(np.min(img2_Xcorners_trans[1])).astype(int)
    max_x = np.ceil(np.max(img2_Xcorners_trans[0])).astype(int)
    max_y = np.ceil(np.max(img2_Xcorners_trans[1])).astype(int)
    
    neg_padding = -1 * min(np.array([0, min_y, min_x])).astype(int)
    pos_padding = max(np.array([max_y-h, max_x-w])).astype(int)
    
    num_colors = image1.shape[2]
    h1 = image1.shape[0]
    w1 = image1.shape[1]
    # create final mosaic image
    mosaic = np.zeros((neg_padding + h1 + pos_padding, neg_padding + w1 + pos_padding, num_colors)).astype(np.uint8)
    # fill the final mosaic image with the reference image(middle)
    mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = image1
    
    # calculate translation homography matrix
    translation = np.array([[1, 0, neg_padding],[0, 1, neg_padding],[0, 0, 1],])
    homography_trans = np.dot(translation, homography_inv)
    
    # transform image2 to mosiac image coordinate
    print("Warping left image...")
    warped_image = np.zeros(mosaic.shape).astype(np.uint8)
    warped_image = cv2.warpPerspective(image2, homography_trans, (warped_image.shape[1], warped_image.shape[0]))
    
    print("Combining two images...")
    
    mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = result
    
    nonzero_img1_inWarped = (np.sum(warped_image, axis=2) > 0)
    nonzero_img2_inMosaic = (np.sum(mosaic, axis=2) > 0)
    
    for r in range(mosaic.shape[0]):
        for c in range(mosaic.shape[1]):
            if nonzero_img2_inMosaic[r, c] and nonzero_img1_inWarped[r, c]:
                mosaic[r,c,:] = mosaic[r,c,:]
            elif nonzero_img1_inWarped[r, c]:
                mosaic[r,c,:] = warped_image[r,c,:]
    return mosaic

if __name__ == "__main__":
    # Philly images by default
    # Switch to Penn images by using left.jpg, middle.jpg and right.jpg
    img1=np.array(Image.open('left2.jpg').convert('RGB'))
    img2=np.array(Image.open('middle2.jpg').convert('RGB'))
    img3=np.array(Image.open('right2.jpg').convert('RGB'))
    #250, 750 for image set2, 500, 3500 for image set1
    w_l = 250 # left carving distance from the left side of the image
    w_r = 750  # right carving distance from the left side of the image
    
    # apply left and right seam carving before left and right mosaic
    seampic1=usingseam_l(w_l,img2)
    seampic=usingseam_r(w_r,seampic1)
    result,result_or=mosaic_l(img2, img1,seampic)
    result=mosaic_r(result_or, img3,result)
    
    plt.imshow(result)
    plt.savefig('result.jpg')
    plt.show()
