# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:48:53 2019

@author: TANG VE
"""
import cv2
import os

import numpy as np
import skvideo.io

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

def drawbox(corners, img):
    bbox_img = img
    x, y, w, h = cv2.boundingRect(corners.astype(int))
    cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return bbox_img

def objectTracking(filename,bboxs):
    cap=cv2.VideoCapture(filename)
    img1=None
    img2=None
    writer=skvideo.io.FFmpegWriter('easyout.avi')
    count=0
    ptsX = np.zeros((1, bboxs.shape[0]))
    ptsY = np.zeros((1, bboxs.shape[0]))
    #k=1
    while(cap.isOpened()):
        count+=1
        ret,frame=cap.read()
        size=7
        if not ret:
            break
        if img1 is None and img2 is None:
            img2=frame
            img2=cv2.GaussianBlur(img2,(size,size),0)
            gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            startYs,startXs=getFeatures(gray,bboxs)
            continue
        '''
        if count==180:
            size=3
            img2=cv2.GaussianBlur(img2,(size,size),0)
            gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            startYs,startXs=getFeatures(gray,bboxs)
            k+=1
            continue
            '''
        img1=img2
        img2=frame
        h,w,rgb=frame.shape
        img2=cv2.GaussianBlur(img2,(size,size),0)
        
        newXs,newYs=estimateAllTranslation(startXs,startYs,img1,img2)
        startXs,startYs,bboxs=applyGeometricTransformation(startXs,startYs,newXs,newYs,bboxs)
        
        bbox_img=frame
        num_boxs=bboxs.shape[0]
        mask=np.ones(num_boxs, dtype=bool)
        for i, bbox in enumerate(bboxs):
            index=np.logical_or(startXs[:,i]>= w,startYs[:,i]>= h)
            startXs[:,i][index]=-1
            startYs[:,i][index]=-1
            temp1=np.int32(startXs[:, i] < 0)
            temp2=np.int32(startYs[:, i] < 0)
            if temp1.min()==1 and temp2.min()==1:
                mask[i] = False
                continue
            bbox_img = drawbox(bbox, bbox_img)

        bboxs=bboxs[mask,:,:]
        startXs=startXs[:,mask]
        startYs=startYs[:,mask]
        ptsX = ptsX[:,mask]
        ptsY = ptsY[:, mask]
        ptsX = np.append(ptsX, startXs, axis=0)
        ptsY = np.append(ptsY, startYs, axis=0)
        if count == 1:
            ptsX = np.delete(ptsX, (0), axis=0)
            ptsY = np.delete(ptsY, (0), axis=0)

        for idx, (x, y) in enumerate(zip(ptsX, ptsY)):
        #for num, (x,y) in enumerate(zip(startXs, startYs)):
            for i in range(bboxs.shape[0]):
                if x[i]>=0 and y[i]>=0:
                    cv2.circle(bbox_img,(np.int32(x[i]),np.int32(y[i])),3,(0,0,255),-1)
        writer.writeFrame(bbox_img[:,:,[2,1,0]])
        cv2.imshow('frame', bbox_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.close()
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # for medium.mp4 video object coordinates
    # bboxs = np.array([[[454,185],[527,185],[454,279],[527,279]]])
    # for easy.mp4 video object coordinates
    bboxs = np.array([[[290, 185], [400, 185], [290, 265], [400, 265]], [[223, 166], [275, 166], [275, 124],[223, 124]]])
    objectTracking('Easy.mp4',bboxs)
