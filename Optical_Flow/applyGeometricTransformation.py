# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:30:39 2019

@author: TANG VE
"""

import numpy as np
import skimage.transform as tf
import random

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    N,F=startXs.shape
    Xs=-np.ones((N, F))
    Ys=-np.ones((N, F))
    newbbox=np.empty((F, 4, 2))
    for i in range(F):
        mask1=np.logical_or(startXs[:,i]>= 0, startYs[:,i]>=0)
        mask2=np.logical_or(newXs[:,i] >= 0, newYs[:,i] >= 0)
        startX=startXs[:,i][mask1]
        startY=startYs[:,i][mask1]
        newX=newXs[:,i][mask2]
        newY=newYs[:,i][mask2]
        
        
        max_inliers=-np.inf
        transformation=None
        N=startX.shape[0]
        pre_corr=np.block([[startX.reshape(1,-1)], [startY.reshape(1,-1)], [np.ones((1, N))]])
        new_dots=np.stack((newX, newY), axis=-1)
        threshold=1

        for k in range(100):
            if N < 3:
                Xs[:newX.size, i]=newX
                Ys[:newY.size, i]=newY
                newbbox[i,:,:]=bbox[i,:]
                break
            index=random.sample(range(N), 3)
            sample_pre=np.stack((startX[index], startY[index]), axis=-1)
            sample_new=np.stack((newX[index], newY[index]), axis=-1)
            transf=tf.SimilarityTransform()
            transf.estimate(sample_pre, sample_new)
            T=np.array(transf.params)
            xs=np.dot(T, pre_corr)[0,:]
            ys=np.dot(T, pre_corr)[1,:]
            comp1=np.block([[xs], [ys]])
            errors=np.sqrt(np.sum((np.transpose(new_dots)-comp1)**2, axis=0))
            comp=np.ones((1,N))*threshold
            temp=np.int32(comp>errors)
            num=temp.sum()
            if max_inliers<num:
                max_inliers=num
                transformation=T
        if N>=3:
            xs=np.dot(transformation, pre_corr)[0,:]
            ys=np.dot(transformation, pre_corr)[1,:]
            comp2=np.block([[xs], [ys]])
            errors=np.sqrt(np.sum((np.transpose(new_dots) - comp2)**2, axis=0))
            xs[errors>threshold]=-1
            ys[errors>threshold]=-1
            temp_bbox=np.dot(transformation, np.block([[np.transpose(bbox[i,:])], [np.ones((1, 4))]]))
            temp_bbox=temp_bbox[[0,1],:]
            temp_bbox=np.transpose(temp_bbox)
            newbbox[i,:,:]=temp_bbox
            Xs[:xs.size,i]=xs
            Ys[:ys.size,i]=ys
    return Xs, Ys, newbbox     
        