'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2

def corner_detector(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
    return kp


    
