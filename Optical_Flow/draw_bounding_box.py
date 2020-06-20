# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:15:09 2019

@author: TANG VE
"""

import cv2

def draw_bounding_box(bb_corners, img):
    bb_img = img
    x, y, w, h = cv2.boundingRect(bb_corners.astype(int))
    cv2.rectangle(bb_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return bb_img